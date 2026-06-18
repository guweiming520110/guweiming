#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钉钉机器人通知模块
推送预测结果、验证结果、准确率统计到钉钉群
"""

import json
import time
import hmac
import hashlib
import base64
import urllib.request
import urllib.parse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

# ==================== 配置 ====================

WEBHOOK_URL = (
    "https://oapi.dingtalk.com/robot/send"
    "?access_token=e41fbb639c62f85b3366168195a5e6e1f546080318cbd9fd6801ccbdb8ddd45d"
)
SECRET = "SECb6a3cf1380e75c515abcb074b3ff5d56877372d2d28bddb862e12e71d973b757"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'

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


def send_dingtalk_message(payload: dict) -> bool:
    """
    发送消息到钉钉

    参数:
        payload: 钉钉消息体 (dict)

    返回:
        bool 是否发送成功
    """
    timestamp = int(time.time() * 1000)
    url = _build_url(timestamp)

    data = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
    }

    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            if result.get('errcode') == 0:
                return True
            else:
                print(f"  钉钉发送失败: {result}")
                return False
    except Exception as e:
        print(f"  钉钉请求异常: {e}")
        return False


def send_text(content: str, at_mobiles: list = None, is_at_all: bool = False) -> bool:
    """发送 text 类型消息"""
    payload = {
        "msgtype": "text",
        "text": {"content": content},
        "at": {"atMobiles": at_mobiles or [], "isAtAll": is_at_all},
    }
    return send_dingtalk_message(payload)


def send_markdown(title: str, text: str, at_mobiles: list = None, is_at_all: bool = False) -> bool:
    """发送 markdown 类型消息"""
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
                               front_report="", back_report="", df=None) -> str:
    """
    格式化预测结果消息

    参数:
        lottery_type: 彩种 (dlt/ssq/kl8)
        front_preds: 前区预测列表 [(num, score, detail), ...]
        back_preds: 后区预测列表 (可选)
        front_report: 前区报告文本
        back_report: 后区报告文本
        df: 数据 DataFrame (用于提取最新期号)

    返回:
        markdown 格式消息字符串
    """
    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    emoji = LOTTERY_EMOJIS.get(lottery_type, '📊')
    period = _get_period_label(lottery_type, df)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# {emoji} {name} · 预测报告", f"", f"**生成时间**: {now_str}"]
    if period != "?":
        lines.append(f"**参考最新期号**: {period}")
    lines.append("")

    if lottery_type == 'kl8':
        # 快乐8：只有前区（20个号码）
        lines.append(f"### 📋 预测前 20 码")
        for rank, (num, score, detail) in enumerate(front_preds[:20], 1):
            # 提取主要图案
            pattern_str = _summarize_patterns(detail)
            bar = '⭐' * min(int(score / 5) + 1, 5)
            lines.append(f"{rank}. **{num}**  `评分:{score:.1f}` {bar}")
            if pattern_str:
                lines.append(f"   └ {pattern_str}")
        lines.append("")
    else:
        # dlt/ssq：前区 + 后区
        front_cnt = 5 if lottery_type == 'dlt' else 6
        back_cnt = 2 if lottery_type == 'dlt' else 1
        front_name = '前区' if lottery_type == 'dlt' else '红球'
        back_name = '后区' if lottery_type == 'dlt' else '蓝球'

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


def format_verification_message(lottery_type: str, predicted_front, predicted_back,
                                 verified_front, verified_back,
                                 actual_front, actual_back) -> str:
    """
    格式化验证结果消息

    参数:
        lottery_type: 彩种
        predicted_front: 预测前区号码列表
        predicted_back: 预测后区号码列表
        verified_front: 命中的前区号码集合
        verified_back: 命中的后区号码集合
        actual_front: 实际前区开奖号码
        actual_back: 实际后区开奖号码
    """
    name = LOTTERY_NAMES.get(lottery_type, lottery_type)
    emoji = LOTTERY_EMOJIS.get(lottery_type, '📊')
    front_name = '前区' if lottery_type == 'dlt' else '红球'
    back_name = '后区' if lottery_type == 'dlt' else '蓝球'
    front_cnt = 5 if lottery_type == 'dlt' else 6

    front_hit = len(verified_front)
    back_hit = len(verified_back or [])
    front_rate = front_hit / front_cnt * 100
    total_hit = front_hit + back_hit
    total_cnt = front_cnt + (2 if lottery_type == 'dlt' else 1)
    total_rate = total_hit / total_cnt * 100

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [f"# {emoji} {name} · 验证结果", "", f"**验证时间**: {now_str}", ""]

    # 预测 vs 实际
    lines.append(f"### 📋 预测 vs 开奖")
    lines.append(f"| | 预测号码 | 实际开奖 | 命中 |")
    lines.append(f"|:--|:--|:--|:--:|")
    pred_f_str = ", ".join(str(n) for n in predicted_front)
    act_f_str = ", ".join(str(n) for n in actual_front)
    hit_f_str = "✅" if front_hit > 0 else "❌"
    lines.append(f"| **{front_name}** | {pred_f_str} | {act_f_str} | {hit_f_str} `{front_hit}/{front_cnt}` |")

    if predicted_back is not None and actual_back is not None:
        pred_b_str = ", ".join(str(n) for n in predicted_back)
        act_b_str = ", ".join(str(n) for n in actual_back)
        hit_b_str = "✅" if back_hit > 0 else "❌"
        lines.append(f"| **{back_name}** | {pred_b_str} | {act_b_str} | {hit_b_str} `{back_hit}/{len(actual_back)}` |")

    lines.append("")

    # 命中详情
    if verified_front:
        lines.append(f"✅ **命中{front_name}**: `{sorted(verified_front)}`")
    if verified_back:
        lines.append(f"✅ **命中{back_name}**: `{sorted(verified_back)}`")
    if not verified_front and not verified_back:
        lines.append(f"❌ 本期**未命中**任何号码")

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


def _summarize_patterns(detail: dict) -> str:
    """从预测详情中提取主要图案摘要"""
    if not detail:
        return ""
    sorted_pats = sorted(detail.items(), key=lambda x: x[1].get('weighted', 0), reverse=True)[:3]
    parts = []
    for label, info in sorted_pats:
        w = info.get('weighted', 0)
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


def _get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn=None):
    """确保 prediction_records 表存在"""
    should_close = conn is None
    if conn is None:
        conn = _get_conn()
    conn.execute("""
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
    """)
    conn.commit()
    if should_close:
        conn.close()


def save_prediction(lottery_type: str, target_period: str,
                    front_numbers: list, back_numbers: list = None) -> int:
    """
    保存预测记录到数据库

    返回:
        新记录的 id
    """
    _ensure_table()
    conn = _get_conn()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    front_str = ','.join(str(n) for n in front_numbers)
    back_str = ','.join(str(n) for n in (back_numbers or []))

    conn.execute(
        """INSERT INTO prediction_records 
           (lottery_type, predict_time, target_period, play_type,
            front_numbers, back_numbers)
           VALUES (?, ?, ?, '几何图案预测', ?, ?)""",
        (lottery_type, now_str, target_period, front_str, back_str)
    )
    conn.commit()
    record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return record_id


def save_verification(lottery_type: str, target_period: str,
                      front_numbers: str, back_numbers: str,
                      actual_front: list, actual_back: list,
                      front_match: set, back_match: set) -> bool:
    """
    更新验证结果到对应的预测记录

    搜索同一彩种+期号+预测号码匹配的记录
    """
    _ensure_table()
    conn = _get_conn()

    front_cnt = 5 if lottery_type == 'dlt' else 6
    actual_f_str = ','.join(str(n) for n in actual_front)
    actual_b_str = ','.join(str(n) for n in actual_back)

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
               front_match_count=?, back_match_count=?
               WHERE id=?""",
            (actual_f_str, actual_b_str,
             1 if front_match or back_match else 0,
             len(front_match), len(back_match),
             row['id'])
        )
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def get_pending_verifications() -> list:
    """获取尚未验证的预测记录"""
    _ensure_table()
    conn = _get_conn()
    cur = conn.execute(
        """SELECT * FROM prediction_records 
           WHERE (actual_front IS NULL OR actual_front = '')
           ORDER BY predict_time DESC"""
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_accuracy_stats() -> dict:
    """
    获取所有彩种的累计准确率统计

    返回:
        { 'dlt': {...}, 'ssq': {...}, 'kl8': {...} }
    """
    _ensure_table()
    conn = _get_conn()
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

    conn.close()
    return stats


# ==================== 发送快捷方法 ====================


def send_prediction(lottery_type: str, front_preds, back_preds=None,
                    front_report="", back_report="", df=None) -> bool:
    """
    发送预测结果到钉钉

    返回:
        bool 是否发送成功
    """
    message = format_prediction_message(
        lottery_type, front_preds, back_preds,
        front_report, back_report, df
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
        f"- 测试时间: {now_str}\n"
        f"- 系统路径: {BASE_DIR}\n\n"
        "---\n"
        "*如果收到此消息，说明钉钉推送配置正确 ✅*"
    )
    return send_markdown("🧪 钉钉推送测试", msg)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='钉钉通知模块')
    parser.add_argument('--test', action='store_true', help='测试钉钉连接')
    parser.add_argument('--report', action='store_true', help='发送统计报告')
    args = parser.parse_args()

    if args.test:
        print("测试钉钉连接...")
        ok = test_connection()
        print(f"{'✅ 发送成功' if ok else '❌ 发送失败'}")

    if args.report:
        print("发送统计报告...")
        ok = send_stats_report()
        print(f"{'✅ 发送成功' if ok else '❌ 发送失败'}")
