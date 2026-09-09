#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钉钉机器人通知模块
推送预测结果、验证结果、准确率统计到钉钉群
"""

import re
import json
import time
import hmac
import hashlib
import base64
import urllib.request
import urllib.parse
import sqlite3
from datetime import datetime
from typing import Optional

from db_utils import get_connection
from config import (
    DINGTALK_TOKEN, DINGTALK_SECRET, DINGTALK_TIMEOUT, DINGTALK_MAX_RETRIES,
    DINGTALK_RATE_LIMIT, DINGTALK_RATE_WINDOW,
)
from logger import log

# ==================== 配置 ====================

# 钉钉凭据统一从 config 读取（config 已加载 .env / 系统环境变量）：
#   DINGTALK_TOKEN  —— 机器人 access_token
#   DINGTALK_SECRET —— 机器人加签 SECRET
# 部署时通过 .env 文件或系统环境变量注入，参见项目根目录 .env.example
_ACCESS_TOKEN = DINGTALK_TOKEN
SECRET = DINGTALK_SECRET

# 调试：打印凭据摘要（仅首次加载时）
if not _ACCESS_TOKEN or not SECRET:
    import warnings
    warnings.warn("钉钉凭据未配置，请设置环境变量 DINGTALK_TOKEN 和 DINGTALK_SECRET")
else:
    pass  # 凭据已配置，不打印摘要

WEBHOOK_URL = (
    "https://oapi.dingtalk.com/robot/send"
    f"?access_token={_ACCESS_TOKEN}"
)

# DB_PATH 统一取自 config（历史路径，不再在此重复定义）

# ==================== 钉钉签名与发送 ====================


def _sign(timestamp: int) -> str:
    """HMAC-SHA256 签名"""
    secret_enc = SECRET.encode('utf-8')
    string_to_sign = f"{timestamp}\n{SECRET}"
    sign = base64.b64encode(
        hmac.new(secret_enc, string_to_sign.encode('utf-8'), hashlib.sha256).digest()
    ).decode('utf-8')
    return sign


def _build_url(timestamp: int) -> str:
    """构建带签名的请求 URL"""
    sign = _sign(timestamp)
    return f"{WEBHOOK_URL}&timestamp={timestamp}&sign={urllib.parse.quote(sign)}"


# 最近一次发送内容缓存（防重复）。注意：只在确认送达成功后才写入，
# 否则一次网络失败会让后续重试被去重逻辑误判为"已发送成功"。
_last_send_cache = {'time': 0, 'content': ''}

# 最近发送时间戳，用于钉钉 20 条/分钟的机器人限流
_send_times = []


def _dedup_key(payload: dict) -> str:
    """生成去重键。

    直接 str(payload) 会因消息里含 datetime.now() 生成的时间戳而永远不相等，
    导致去重完全失效；这里把时间串统一替换为占位符。
    """
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?', '<TS>', s)


def _wait_for_rate_limit():
    """钉钉机器人限流：同一机器人每分钟最多 20 条，超出则等待窗口滑出。"""
    global _send_times
    now = time.time()
    _send_times = [t for t in _send_times if now - t < DINGTALK_RATE_WINDOW]
    if len(_send_times) >= DINGTALK_RATE_LIMIT:
        wait = DINGTALK_RATE_WINDOW - (now - _send_times[0]) + 1
        if wait > 0:
            log.warning("钉钉限流：每分钟最多 %d 条，等待 %.0f 秒",
                        DINGTALK_RATE_LIMIT, wait)
            time.sleep(wait)
        now = time.time()
        _send_times = [t for t in _send_times if now - t < DINGTALK_RATE_WINDOW]


def send_dingtalk_message(payload: dict) -> bool:
    """
    发送消息到钉钉（带 10 秒去重、限流等待、错误分类与重试）

    参数:
        payload: 钉钉消息体 (dict)

    返回:
        bool 是否发送成功（去重命中返回 True，因为上一次确实已送达）
    """
    # 空凭据快速失败：未配置 token/secret 时不发起网络请求，也不做无谓的
    # 3 次重试 + 限流等待，直接返回失败（查询/统计等功能不受影响）。
    if not _ACCESS_TOKEN or not SECRET:
        log.warning("钉钉未配置 token/secret，跳过推送")
        return False

    now = time.time()
    dedup_key = _dedup_key(payload)

    # 10 秒内相同内容不发第二次
    if now - _last_send_cache['time'] < 10 and _last_send_cache['content'] == dedup_key:
        print("  ⏭️ 钉钉去重: 10秒内相同消息已发送，跳过")
        return True

    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    for attempt in range(DINGTALK_MAX_RETRIES):
        _wait_for_rate_limit()
        timestamp = int(time.time() * 1000)
        req = urllib.request.Request(_build_url(timestamp), data=data,
                                     headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=DINGTALK_TIMEOUT) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            errcode = result.get('errcode')
            if errcode == 0:
                # 仅在确认送达后写缓存
                _last_send_cache['time'] = time.time()
                _last_send_cache['content'] = dedup_key
                _send_times.append(time.time())
                return True
            if errcode == 130101:
                # 发送过快/限流，退避后重试
                log.warning("钉钉限流(errcode=130101)，60 秒后重试")
                time.sleep(60)
                continue
            if errcode == 310000:
                log.error("钉钉加签校验失败(310000)，请检查 DINGTALK_SECRET: %s", result)
                return False
            if errcode == 300001:
                log.error("钉钉 access_token 无效(300001)，请检查 DINGTALK_TOKEN: %s", result)
                return False
            log.error("钉钉发送失败: %s", result)
            return False
        except Exception as e:
            log.warning("钉钉请求异常(第%d/%d次): %s",
                        attempt + 1, DINGTALK_MAX_RETRIES, e)
            if attempt < DINGTALK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return False


def send_text(content: str, at_mobiles: list | None = None, is_at_all: bool = False) -> bool:
    """发送 text 类型消息"""
    payload = {
        "msgtype": "text",
        "text": {"content": content},
        "at": {"atMobiles": at_mobiles or [], "isAtAll": is_at_all},
    }
    return send_dingtalk_message(payload)


def send_markdown(title: str, text: str, at_mobiles: list | None = None, is_at_all: bool = False) -> bool:
    """发送 markdown 类型消息
    钉钉机器人限制：消息内容不能超过 5000 字符，超出会自动截断
    """
    # 钉钉限制消息长度 5000 字符，超出截断防止发送失败
    MAX_LENGTH = 4500  # 留一些余量给标题和包装
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH] + "\n\n... (内容过长已截断)"
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": text},
        "at": {"atMobiles": at_mobiles or [], "isAtAll": is_at_all},
    }
    return send_dingtalk_message(payload)


# ==================== 消息格式化 ====================


LOTTERY_NAMES = {
    'dlt': '超级大乐透',
    'ssq': '双色球',
    'kl8': '快乐8',
    'dlt_back': '大乐透后区',
    'ssq_back': '双色球蓝球',
}

LOTTERY_EMOJIS = {
    'dlt': '🎯',
    'ssq': '🔴',
    'kl8': '🎲',
}


def _get_period_label(lottery_type: str, df=None) -> str:
    """获取彩种的最新期号标签"""
    if df is not None and len(df) > 0:
        return str(df.iloc[-1]['period'])
    return "?"


def format_prediction_message(lottery_type: str, front_preds, back_preds=None,
                               front_report="", back_report="", df=None,
                               kl8_combos=None, target_period=None) -> str:
    """
    格式化预测结果消息

    参数:
        lottery_type: 彩种 (dlt/ssq/kl8)
        front_preds: 前区预测列表 [(num, score, detail), ...]
        back_preds: 后区预测列表 (可选)
        front_report: 前区报告文本
        back_report: 后区报告文本
        df: 数据 DataFrame (用于提取最新期号)
        kl8_combos: 快乐8投注组合 (generate_kl8_combinations 返回值)
        target_period: 本次预测的目标期号（下一期，None 时不显示）

    返回:
        markdown 格式消息字符串
    """
    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    emoji = LOTTERY_EMOJIS.get(lottery_type, '📊')
    period = _get_period_label(lottery_type, df)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# {emoji} {name} · 预测报告", f"", f"**生成时间**: {now_str}"]
    # 优先突出"预测目标期号"（下一期），避免误解为在预测已开奖的最新期
    if target_period:
        lines.append(f"**🎯 预测目标期号**: {target_period}（下一期）")
    if period != "?":
        lines.append(f"**参考最新开奖期号**: {period}（已开奖）")
    lines.append("")

    # 前/后区配置与彩种无关地先赋值，供后续汇总分支安全引用（kl8 不会用到后区值）
    front_cnt = 5 if lottery_type == 'dlt' else 6
    back_cnt = 2 if lottery_type == 'dlt' else 1
    front_name = '前区' if lottery_type == 'dlt' else '红球'
    back_name = '后区' if lottery_type == 'dlt' else '蓝球'
    if lottery_type == 'kl8':
        _format_kl8_prediction_message(lines, front_preds, kl8_combos)
    else:
        # dlt/ssq：前区 + 后区
        lines.append(f"### 📋 预测{front_name} Top-{front_cnt}")
        for rank, (num, score, detail) in enumerate(front_preds[:front_cnt], 1):
            pattern_str = _summarize_patterns(detail)
            bar = '⭐' * min(int(score / 3) + 1, 5)
            lines.append(f"{rank}. **{num}**  `评分:{score:.1f}` {bar}")
            if pattern_str:
                lines.append(f"   └ {pattern_str}")
        lines.append("")

        if back_preds:
            lines.append(f"### 📋 预测{back_name} Top-{back_cnt}")
            for rank, (num, score, detail) in enumerate(back_preds[:back_cnt], 1):
                pattern_str = _summarize_patterns(detail)
                lines.append(f"{rank}. **{num}**  `评分:{score:.1f}`")
                if pattern_str:
                    lines.append(f"   └ {pattern_str}")
            lines.append("")

    # 提取回测统计（如果有）
    if front_report:
        bt_lines = _extract_backtest_from_report(front_report)
        if bt_lines:
            lines.append("### 📊 回测统计")
            lines.extend(bt_lines)
            lines.append("")

    # 汇总
    if lottery_type == 'kl8':
        if kl8_combos:
            from prize_calculator import _fmt_nums
            nums_str = (
                f"单式A: `{_fmt_nums(kl8_combos['single'][0])}`\n\n"
                f"单式B: `{_fmt_nums(kl8_combos['single'][1])}`\n\n"
                f"复式11码①: `{_fmt_nums(kl8_combos['complex_11'][0])}`\n\n"
                f"复式11码②: `{_fmt_nums(kl8_combos['complex_11'][1])}`\n\n"
                f"复式12码①: `{_fmt_nums(kl8_combos['complex_12'][0])}`\n\n"
                f"复式12码②: `{_fmt_nums(kl8_combos['complex_12'][1])}`"
            )
        else:
            nums_str = ", ".join(str(n) for n, s, d in front_preds[:20])
    else:
        front_nums = [str(n) for n, s, d in front_preds[:front_cnt]]
        back_nums = [str(n) for n, s, d in (back_preds or [])[:back_cnt]]
        nums_str = f"{front_name}: {' '.join(front_nums)} | {back_name}: {' '.join(back_nums)}"

    lines.append(f"> **预测号码**: {nums_str}")
    lines.append("")
    lines.append("---")
    lines.append(f"*🤖 由数字彩票AI分析系统自动生成*")

    return "\n".join(lines)


def format_verification_message(lottery_type: str, predicted_front: list, predicted_back: Optional[list],
                                 verified_front: set, verified_back: Optional[set],
                                 actual_front: list, actual_back: Optional[list]) -> str:
    """
    格式化验证结果消息（含命中详情和中奖详情）

    参数:
        lottery_type: 彩种
        predicted_front: 预测前区号码列表
        predicted_back: 预测后区号码列表
        verified_front: 命中的前区号码集合
        verified_back: 命中的后区号码集合
        actual_front: 实际前区开奖号码
        actual_back: 实际后区开奖号码
    """
    from prize_calculator import (format_hit_detail, format_prize_detail,
                                   get_prize, _fmt_nums)

    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    emoji = LOTTERY_EMOJIS.get(lottery_type, '📊')

    if lottery_type == 'kl8':
        front_name = '号码'
        front_cnt = 20
        back_name = ''  # kl8 无后区，verify 分支已用 `if lottery_type != 'kl8'` 守卫
        front_hit = len(verified_front)
        back_hit = 0
        total_hit = front_hit
        total_cnt = 20
        total_rate = front_hit / front_cnt * 100
    else:
        front_name = '前区' if lottery_type == 'dlt' else '红球'
        back_name = '后区' if lottery_type == 'dlt' else '蓝球'
        front_cnt = 5 if lottery_type == 'dlt' else 6
        back_cnt = 2 if lottery_type == 'dlt' else 1
        front_hit = len(verified_front)
        verified_back = verified_back or set()
        back_hit = len(verified_back)
        total_hit = front_hit + back_hit
        total_cnt = front_cnt + (2 if lottery_type == 'dlt' else 1)
        total_rate = total_hit / total_cnt * 100
        if actual_back is None:
            actual_back = []
        if predicted_back is None:
            predicted_back = []

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# {emoji} {name} · 验证结果", "", f"**验证时间**: {now_str}", ""]

    # ── 预测 vs 开奖对照表 ──
    lines.append(f"### 📋 预测 vs 开奖")
    lines.append(f"| | 预测号码 | 实际开奖 | 命中 |")
    lines.append(f"|:--|:--|:--|:--:|")
    pred_f_str = _fmt_nums(predicted_front)
    act_f_str = _fmt_nums(actual_front)
    hit_f_str = "✅" if front_hit > 0 else "❌"
    lines.append(f"| **{front_name}** | {pred_f_str} | {act_f_str} | {hit_f_str} `{front_hit}/{front_cnt}` |")

    if lottery_type != 'kl8' and predicted_back and actual_back:
        pred_b_str = _fmt_nums(predicted_back)
        act_b_str = _fmt_nums(actual_back)
        hit_b_str = "✅" if back_hit > 0 else "❌"
        lines.append(f"| **{back_name}** | {pred_b_str} | {act_b_str} | {hit_b_str} `{back_hit}/{len(actual_back)}` |")

    lines.append("")

    # ── 命中详情（命中/未命中逐一列出）──
    lines.append(f"### 🎯 命中详情")
    lines.append(format_hit_detail(
        lottery_type, predicted_front, predicted_back or [],
        actual_front, actual_back or []
    ))
    lines.append("")

    # ── 中奖详情（奖级、奖金）──
    lines.append(f"### 🏆 中奖详情")
    prize_result = get_prize(lottery_type, front_hit, back_hit)
    lines.append(format_prize_detail(lottery_type, front_hit, back_hit))

    if prize_result.get('is_winning'):
        lines.append(f"")
        lines.append(f"> 🎉 **恭喜中奖！{prize_result.get('level', '')}**")
    elif lottery_type == 'kl8' and prize_result.get('match_rate'):
        # 旧写法里 is_winning 的定义就是 match_count>=10，导致该 elif 永远不可达。
        # 且 kl8 预测的是 20 个号码，"命中 N 个"并不等于某一注选十中奖，
        # 因此这里只给命中率评价，真实中奖由 send_kl8_verification 推送。
        lines.append(f"")
        lines.append(
            f"> 📈 **命中率 {prize_result.get('match_rate')}%"
            f"（{prize_result.get('level', '')}）**"
        )

    lines.append("")
    lines.append(f"> **综合命中率**: {total_hit}/{total_cnt} = **{total_rate:.1f}%**")
    lines.append("")
    lines.append("---")
    lines.append(f"*🤖 由数字彩票AI分析系统自动验证*")

    return "\n".join(lines)


def format_stats_message(stats: dict) -> str:
    """
    格式化历史统计报告

    参数:
        stats: {
            'dlt': {'total': N, 'verified': N, 'front_hit_rate': %, 'back_hit_rate': %, ...},
            'ssq': {...},
            'kl8': {...},
        }
    """
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# 📈 彩票AI预测 · 累计统计报告", "", f"**生成时间**: {now_str}", ""]

    for lt in ['dlt', 'ssq', 'kl8']:
        s = stats.get(lt)
        if not s or s['total'] == 0:
            continue
        name = LOTTERY_NAMES.get(lt, lt)
        emoji = LOTTERY_EMOJIS.get(lt, '📊')

        lines.append(f"### {emoji} {name}")
        lines.append(f"- 总预测次数: **{s['total']}**")
        lines.append(f"- 已验证: **{s['verified']}** 次")
        if s.get('front_hit_rate') is not None:
            lines.append(f"- {s['front_name']}命中率: **{s['front_hit_rate']:.1f}%** ({s['front_hit']}/{s['front_total']})")
        if s.get('back_hit_rate') is not None:
            lines.append(f"- {s['back_name']}命中率: **{s['back_hit_rate']:.1f}%** ({s['back_hit']}/{s['back_total']})")
        if s.get('avg_top10_hit') is not None:
            lines.append(f"- Top10 平均命中: **{s['avg_top10_hit']:.2f}**")
        lines.append("")

    lines.append("---")
    lines.append(f"*🤖 由数字彩票AI分析系统自动统计*")

    return "\n".join(lines)


# ==================== 工具函数 ====================


def _format_kl8_prediction_message(lines: list, front_preds, kl8_combos):
    """格式化快乐8预测消息（选十投注组合格式）"""
    from prize_calculator import _fmt_nums

    if kl8_combos is None:
        # 兼容旧格式：只显示 Top-20
        lines.append(f"### 📋 预测前 20 码")
        for rank, (num, score, detail) in enumerate(front_preds[:20], 1):
            pattern_str = _summarize_patterns(detail)
            bar = '⭐' * min(int(score / 5) + 1, 5)
            lines.append(f"{rank}. **{num}**  `评分:{score:.1f}` {bar}")
            if pattern_str:
                lines.append(f"   └ {pattern_str}")
        lines.append("")
        return

    # ── 单式选十 ──
    lines.append(f"### 🎯 单式选十（2组）")
    labels = ['A', 'B']
    for i, combo in enumerate(kl8_combos['single']):
        lines.append(f"- **{labels[i]}组**: `{_fmt_nums(combo)}`  `[{len(combo)}码]`")
    lines.append("")

    # ── 复式11码 10+1 ──
    lines.append(f"### 📐 复式11码 10+1（2注）")
    for i, combo in enumerate(kl8_combos['complex_11']):
        lines.append(f"- **注{i+1}**: `{_fmt_nums(combo)}`  `[{len(combo)}码]`")
    lines.append("")

    # ── 复式12码 10+2 ──
    lines.append(f"### 📐 复式12码 10+2（2注）")
    for i, combo in enumerate(kl8_combos['complex_12']):
        lines.append(f"- **注{i+1}**: `{_fmt_nums(combo)}`  `[{len(combo)}码]`")
    lines.append("")

    # ── 参考 Top-20 ──
    top_20 = kl8_combos.get('top_20', [])
    lines.append(f"### 📋 参考 Top-20 码")
    lines.append(f"> `{_fmt_nums(top_20)}`")
    lines.append("")


def _summarize_patterns(detail):
    """从预测详情中提取主要图案摘要"""
    if not detail:
        return ""
    # 过滤出值是 dict 的条目（遗漏字符串跳过）
    items = [(k, v) for k, v in detail.items() if isinstance(v, dict)]
    sorted_pats = sorted(items, key=lambda x: x[1].get('weighted', 0), reverse=True)[:3]
    parts = []
    for label, info in sorted_pats:
        w = info.get('weighted', 0)
        if w > 0:
            parts.append(f"{label}({w:.1f})")
    return ", ".join(parts)


def _extract_backtest_from_report(report: str) -> list:
    """从预测报告中提取回测统计部分"""
    lines = []
    in_backtest = False
    for line in report.split('\n'):
        if '回测结果' in line:
            in_backtest = True
            continue
        if in_backtest:
            stripped = line.strip()
            if stripped and not stripped.startswith('='):
                lines.append(stripped)
            if '=' in line and len(stripped) < 10:
                break
    return lines


# ==================== 数据库操作 ====================


# 统一走 db_utils.get_connection() 管理连接（自动 WAL + busy_timeout + 事务），
# 不再用裸 sqlite3.connect。查询功能（_ensure_table/get_accuracy_stats 等）
# 不依赖钉钉凭据，即使未配置 token/secret 也可正常使用。


def _ensure_table(conn=None):
    """确保 prediction_records 表存在；传入 conn 时复用该连接（兼容批量写入场景）"""
    ddl = """
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lottery_type VARCHAR(20) NOT NULL,
            predict_time VARCHAR(30) NOT NULL DEFAULT '',
            target_period VARCHAR(20) NOT NULL DEFAULT '',
            play_type VARCHAR(50) NOT NULL DEFAULT '几何图案预测',
            front_numbers VARCHAR(200) NOT NULL DEFAULT '',
            back_numbers VARCHAR(100) NOT NULL DEFAULT '',
            front_probabilities VARCHAR(1000) NOT NULL DEFAULT '',
            back_probabilities VARCHAR(500) NOT NULL DEFAULT '',
            front_dan VARCHAR(50) NOT NULL DEFAULT '',
            front_tuo VARCHAR(200) NOT NULL DEFAULT '',
            back_dan VARCHAR(20) NOT NULL DEFAULT '',
            back_tuo VARCHAR(50) NOT NULL DEFAULT '',
            combination_count INTEGER NOT NULL DEFAULT 0,
            actual_front VARCHAR(100) NOT NULL DEFAULT '',
            actual_back VARCHAR(50) NOT NULL DEFAULT '',
            is_matched INTEGER NOT NULL DEFAULT 0,
            match_level VARCHAR(20) NOT NULL DEFAULT '',
            front_match_count INTEGER NOT NULL DEFAULT 0,
            back_match_count INTEGER NOT NULL DEFAULT 0
        )
    """
    if conn is not None:
        conn.execute(ddl)
        conn.commit()
        return
    with get_connection() as new_conn:  # 独立连接：退出时自动提交
        new_conn.execute(ddl)


def save_prediction(lottery_type: str, target_period: str,
                    front_numbers: list, back_numbers: list | None = None) -> int:
    """
    保存预测记录到数据库（同彩种+同期号已存在时更新，否则插入）

    返回:
        新记录或已有记录的 id
    """
    with get_connection() as conn:
        _ensure_table(conn)  # 复用同一连接，避免双连接竞争
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        front_str = ','.join(str(n) for n in front_numbers)
        back_str = ','.join(str(n) for n in (back_numbers or []))

        # 先查找是否已存在同一彩种+同期号的记录
        cur = conn.execute(
            """SELECT id FROM prediction_records
               WHERE lottery_type=? AND target_period=?
               ORDER BY id DESC LIMIT 1""",
            (lottery_type, target_period)
        )
        existing = cur.fetchone()

        if existing:
            # 更新已有记录（保留验证信息）
            conn.execute(
                """UPDATE prediction_records SET
                   predict_time=?, front_numbers=?, back_numbers=?, play_type='几何图案预测'
                   WHERE id=?""",
                (now_str, front_str, back_str, existing[0])
            )
            record_id = existing[0]
        else:
            # 插入新记录
            conn.execute(
                """INSERT INTO prediction_records
                   (lottery_type, predict_time, target_period, play_type,
                    front_numbers, back_numbers)
                   VALUES (?, ?, ?, '几何图案预测', ?, ?)""",
                (lottery_type, now_str, target_period, front_str, back_str)
            )
            record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # with 块正常退出时自动 commit
        return record_id


def save_verification(lottery_type: str, target_period: str,
                      front_numbers: str, back_numbers: str,
                      actual_front: list, actual_back: list,
                      front_match: set, back_match: set) -> bool:
    """
    更新验证结果到对应的预测记录（含奖级信息）

    搜索同一彩种+期号+预测号码匹配的记录
    """
    from prize_calculator import get_prize_level_string

    with get_connection(sqlite3.Row) as conn:
        _ensure_table(conn)  # 复用同一连接

        actual_f_str = ','.join(str(n) for n in actual_front)
        actual_b_str = ','.join(str(n) for n in actual_back)

        # 计算奖级
        match_level = get_prize_level_string(
            lottery_type, len(front_match), len(back_match)
        )

        # 查找匹配的预测记录
        cur = conn.execute(
            """SELECT id FROM prediction_records
               WHERE lottery_type=? AND target_period=?
               AND front_numbers=?
               ORDER BY id DESC LIMIT 1""",
            (lottery_type, target_period, front_numbers)
        )
        row = cur.fetchone()

        if row:
            conn.execute(
                """UPDATE prediction_records SET
                   actual_front=?, actual_back=?,
                   is_matched=?,
                   match_level=?,
                   front_match_count=?, back_match_count=?
                   WHERE id=?""",
                (actual_f_str, actual_b_str,
                 1 if front_match or back_match else 0,
                 match_level,
                 len(front_match), len(back_match),
                 row['id'])
            )
            return True  # with 块正常退出时自动 commit
        return False


def get_pending_verifications() -> list:
    """获取尚未验证的预测记录"""
    _ensure_table()
    with get_connection(sqlite3.Row, commit=False) as conn:
        cur = conn.execute(
            """SELECT * FROM prediction_records
               WHERE (actual_front IS NULL OR actual_front = '')
               ORDER BY predict_time DESC"""
        )
        rows = [dict(r) for r in cur.fetchall()]
    return rows


def get_accuracy_stats() -> dict:
    """
    获取所有彩种的累计准确率统计

    返回:
        { 'dlt': {...}, 'ssq': {...}, 'kl8': {...} }
    """
    _ensure_table()
    with get_connection(commit=False) as conn:
        stats = {}

        for lt in ['dlt', 'ssq', 'kl8']:
            cur = conn.execute(
                "SELECT COUNT(*) FROM prediction_records WHERE lottery_type=?",
                (lt,)
            )
            total = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prediction_records WHERE lottery_type=? AND actual_front != ''",
                (lt,)
            )
            verified = cur.fetchone()[0]

            entry = {
                'total': total,
                'verified': verified,
            }

            if lt == 'kl8':
                entry['front_name'] = '号码'
                entry['back_name'] = None
                if verified > 0:
                    cur = conn.execute(
                        """SELECT front_match_count FROM prediction_records
                           WHERE lottery_type=? AND actual_front != ''""",
                        (lt,)
                    )
                    hits = [r[0] for r in cur.fetchall()]
                    entry['front_hit'] = sum(hits)
                    entry['front_total'] = len(hits) * 20  # kl8 draws 20 numbers
                    entry['front_hit_rate'] = (sum(hits) / (len(hits) * 20)) * 100 if hits else 0
                    entry['avg_top10_hit'] = sum(hits) / len(hits) if hits else 0
            else:
                front_cnt = 5 if lt == 'dlt' else 6
                back_cnt = 2 if lt == 'dlt' else 1
                entry['front_name'] = '前区' if lt == 'dlt' else '红球'
                entry['back_name'] = '后区' if lt == 'dlt' else '蓝球'

                if verified > 0:
                    cur = conn.execute(
                        """SELECT front_match_count, back_match_count FROM prediction_records
                           WHERE lottery_type=? AND actual_front != ''""",
                        (lt,)
                    )
                    rows = cur.fetchall()
                    f_hits = sum(r[0] for r in rows)
                    b_hits = sum(r[1] for r in rows)
                    entry['front_hit'] = f_hits
                    entry['front_total'] = len(rows) * front_cnt
                    entry['front_hit_rate'] = (f_hits / (len(rows) * front_cnt)) * 100 if rows else 0
                    entry['back_hit'] = b_hits
                    entry['back_total'] = len(rows) * back_cnt
                    entry['back_hit_rate'] = (b_hits / (len(rows) * back_cnt)) * 100 if rows else 0

            stats[lt] = entry

        return stats


# ==================== 发送快捷方法 ====================


def send_prediction(lottery_type: str, front_preds, back_preds=None,
                    front_report="", back_report="", df=None,
                    kl8_combos=None, target_period=None) -> bool:
    """
    发送预测结果到钉钉

    返回:
        bool 是否发送成功
    """
    message = format_prediction_message(
        lottery_type, front_preds, back_preds,
        front_report, back_report, df, kl8_combos, target_period
    )
    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    title = f"{name} · 预测结果"
    return send_markdown(title, message)


def send_verification(lottery_type: str, predicted_front, predicted_back,
                       verified_front, verified_back,
                       actual_front, actual_back) -> bool:
    """
    发送验证结果到钉钉
    """
    message = format_verification_message(
        lottery_type, predicted_front, predicted_back,
        verified_front, verified_back,
        actual_front, actual_back
    )
    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    title = f"{name} · 验证结果"
    return send_markdown(title, message)


def format_kl8_verification_message(kl8_combos: dict, actual_nums: list,
                                     target_period: str = "") -> str:
    """
    格式化快乐8选十投注组合验证结果消息（精简版）

    参数:
        kl8_combos: generate_kl8_combinations() 返回值
        actual_nums: 实际开奖号码列表
        target_period: 目标期号
    """
    from prize_calculator import (_fmt_nums, verify_kl8_combinations,
                                   get_kl8_sh10_prize)

    verified = verify_kl8_combinations(kl8_combos, actual_nums)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# 🎲 快乐8 · 验证结果", "", f"**验证时间**: {now_str}"]
    if target_period:
        lines.append(f"**目标期号**: {target_period}")
    lines.append("")

    # ── Top-20 整体命中（放在最前面，最重要）──
    top_20 = kl8_combos.get('top_20', [])
    top_20_hit = verified['top_20_hit']
    hit_count = verified['top_20_hit_count']
    hit_rate = hit_count / 20 * 100 if top_20 else 0

    lines.append(f"### 📊 Top-20 整体命中")
    lines.append(f"- **预测**: `{_fmt_nums(sorted(top_20))}`")
    lines.append(f"- **开奖**: `{_fmt_nums(sorted(actual_nums))}`")
    lines.append(f"- **命中**: `{_fmt_nums(sorted(top_20_hit))}` ({hit_count}/20 = {hit_rate:.1f}%)")
    if top_20 and hit_count < 20:
        miss = sorted(set(top_20) - set(actual_nums))
        lines.append(f"- 未命中: `{_fmt_nums(miss)}`")
    lines.append("")

    # ── 各组合中奖汇总（精简表格）──
    lines.append(f"### 📋 组合验证结果")
    lines.append(f"| 类型 | 组号 | 命中 | 结果 |")
    lines.append(f"|:--|:--:|:--:|:--|")

    labels = ['A', 'B']
    for i, v in enumerate(verified['single']):
        result = v['prize']
        if result and result['is_winning']:
            res_str = f"🎉 {result['level']} {result['bonus']}"
        elif result:
            res_str = result['match_desc']
        else:
            res_str = "-"
        lines.append(f"| 单式 | {labels[i]} | {v['hit_count']}/10 | {res_str} |")

    for i, v in enumerate(verified['complex_11']):
        result = v['prize']
        if result and result['is_winning']:
            res_str = f"🎉 {result['level']} {result['bonus']}"
        elif result:
            res_str = result['match_desc']
        else:
            res_str = "-"
        lines.append(f"| 11码 | {i+1} | {v['hit_count']}/10 | {res_str} |")

    for i, v in enumerate(verified['complex_12']):
        result = v['prize']
        if result and result['is_winning']:
            res_str = f"🎉 {result['level']} {result['bonus']}"
        elif result:
            res_str = result['match_desc']
        else:
            res_str = "-"
        lines.append(f"| 12码 | {i+1} | {v['hit_count']}/10 | {res_str} |")
    lines.append("")

    # ── 汇总 ──
    any_winning = False
    best_prize = None
    for key in ['single', 'complex_11', 'complex_12']:
        for v in verified[key]:
            if v['prize'] and v['prize']['is_winning']:
                any_winning = True
                if best_prize is None or v['prize']['match_count'] > best_prize['match_count']:
                    best_prize = v['prize']

    if any_winning and best_prize:
        lines.append(f"> 🎉 **恭喜中奖！最高奖级: {best_prize['level']} {best_prize['bonus']}**")
    else:
        lines.append(f"> 📊 本次未中奖，继续加油！")

    lines.append("")
    lines.append("---")
    lines.append(f"*🤖 由数字彩票AI分析系统自动验证*")

    return "\n".join(lines)


def send_kl8_verification(kl8_combos: dict, actual_nums: list,
                           target_period: str = "") -> bool:
    """
    发送快乐8组合验证结果到钉钉
    """
    message = format_kl8_verification_message(kl8_combos, actual_nums, target_period)
    return send_markdown("快乐8 · 验证结果", message)


def send_stats_report() -> bool:
    """
    发送累计统计报告到钉钉
    """
    stats = get_accuracy_stats()
    message = format_stats_message(stats)
    return send_markdown("📈 彩票AI预测 · 累计统计报告", message)


# ==================== 测试 ====================


def test_connection() -> bool:
    """测试钉钉连接是否正常"""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = (
        "## ✅ 钉钉推送测试\n\n"
        f"**数字彩票AI分析系统** 钉钉推送功能连接测试\n\n"
        f"- 测试时间: {now_str}\n\n"
        "---\n"
        "*如果收到此消息，说明钉钉推送配置正确 ✅*"
    )
    return send_markdown("🧪 钉钉推送测试", msg)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='钉钉通知模块')
    parser.add_argument('--test', action='store_true', help='测试钉钉连接')
    parser.add_argument('--report', action='store_true', help='发送统计报告')
    parser.add_argument('--repush-verify', action='store_true', help='重推所有已验证结果到钉钉')
    parser.add_argument('--lottery-type', default=None, help='只重推指定彩种 (dlt/ssq/kl8)')
    args = parser.parse_args()

    if args.test:
        print("测试钉钉连接...")
        ok = test_connection()
        print(f"{'✅ 发送成功' if ok else '❌ 发送失败'}")

    if args.report:
        print("发送统计报告...")
        ok = send_stats_report()
        print(f"{'✅ 发送成功' if ok else '❌ 发送失败'}")

    if args.repush_verify:
        print("开始重推所有验证结果...")
        with get_connection(sqlite3.Row, commit=False) as conn:
            cur = conn.execute(
                """SELECT * FROM prediction_records
                   WHERE actual_front IS NOT NULL AND actual_front != ''
                   ORDER BY target_period DESC"""
            )
            if args.lottery_type:
                # 只重推指定彩种
                cur = conn.execute(
                    """SELECT * FROM prediction_records
                       WHERE actual_front IS NOT NULL AND actual_front != ''
                       AND lottery_type = ?
                       ORDER BY target_period DESC""",
                    (args.lottery_type,)
                )
            rows = [dict(r) for r in cur.fetchall()]
        print(f"找到 {len(rows)} 条已验证记录")
        success = 0
        failed = 0
        for r in rows:
            lt = r['lottery_type']
            target_period = r['target_period']
            front_pred = [int(x) for x in r['front_numbers'].split(',') if x.strip().isdigit()]
            back_pred = [int(x) for x in r['back_numbers'].split(',') if x.strip().isdigit()] if r['back_numbers'] else []
            actual_front = [int(x) for x in r['actual_front'].split(',') if x.strip().isdigit()]
            actual_back = [int(x) for x in r['actual_back'].split(',') if x.strip().isdigit()] if r['actual_back'] else []

            # 计算命中
            front_match = set(front_pred) & set(actual_front)
            back_match = set(back_pred) & set(actual_back)

            print(f"  推送 {lt} 期{target_period}... ", end='', flush=True)
            try:
                if lt == 'kl8':
                    from prize_calculator import generate_kl8_combinations
                    kl8_combos = generate_kl8_combinations(front_pred)
                    ok = send_kl8_verification(kl8_combos, actual_front, target_period)
                else:
                    ok = send_verification(lt, front_pred, back_pred, front_match, back_match, actual_front, actual_back)
                if ok:
                    print("✅ 成功")
                    success += 1
                else:
                    print("❌ 失败")
                    failed += 1
                # 间隔1秒避免钉钉限流
                import time
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ 异常: {e}")
                failed += 1
        print(f"\n重推完成: 成功 {success} 条, 失败 {failed} 条")
