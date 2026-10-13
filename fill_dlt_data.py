#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透数据补全模块
从500彩票网爬取缺失的历史开奖数据
"""

import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup

from config import HTTP_ENCODING, HTTP_MAX_RETRIES, HTTP_TIMEOUT
from db_utils import get_connection
from http_utils import get_with_retry, sleep_interval
from logger import log
from lottery_types import next_period

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def _get_with_retry(url, max_retries=HTTP_MAX_RETRIES, timeout=HTTP_TIMEOUT):
    """向后兼容包装：统一委托 http_utils.get_with_retry（指数退避重试）"""
    return get_with_retry(url, max_retries=max_retries, timeout=timeout,
                          encoding=HTTP_ENCODING)


def get_db_status():
    """检查数据库中大乐透数据的完整状态"""
    with get_connection(commit=False) as conn:
        cur = conn.execute(
            "SELECT COUNT(*), MIN(period), MAX(period) FROM lottery_records WHERE lottery_type='dlt'"
        )
        cnt, pmin, pmax = cur.fetchone()

        cur = conn.execute(
            "SELECT period, date FROM lottery_records WHERE lottery_type='dlt' ORDER BY period"
        )
        rows = cur.fetchall()

    # 非数字期号直接跳过，避免 int() 抛异常中断整个补全流程
    periods = [int(r[0]) for r in rows if str(r[0]).strip().isdigit()]

    gaps = []
    for i in range(len(periods) - 1):
        gap = periods[i + 1] - periods[i]
        if gap <= 1:
            continue
        same_year = str(periods[i])[:2] == str(periods[i + 1])[:2]
        # 跨年场景用 next_period 判断是否属于正常的年度滚动
        # （如 25150 → 26001 属正常，不应算作缺口）
        is_rollover = False
        try:
            is_rollover = next_period('dlt', str(periods[i])) == str(periods[i + 1])
        except Exception:
            is_rollover = False
        # 忽略跨度异常大的情况（历史数据本身缺失过多，区间爬取也补不回来）
        if (same_year or not is_rollover) and (gap - 1) <= 500:
            gaps.append((periods[i], periods[i + 1], gap - 1))

    return {
        'count': cnt,
        'min_period': pmin,
        'max_period': pmax,
        'latest_date': rows[-1][1] if rows else None,
        'gaps': gaps,
        'latest_period': int(pmax) if pmax and str(pmax).strip().isdigit() else 0,
    }


def fetch_dlt_page(start_period, end_period):
    """从500彩票网获取大乐透数据页"""
    url = f'https://datachart.500.com/dlt/history/newinc/history.php?start={start_period}&end={end_period}'
    resp = _get_with_retry(url)
    if resp is None:
        return None
    # 编码已由 http_utils 统一设置（GBK），此处不再硬编码 gb2312
    return resp.text


def _pick_date(texts):
    """从表格单元格文本中挑出合法的 YYYY-MM-DD 日期，找不到返回空串。

    不能直接用最后一列：500.com 表格末列可能是奖池/销售额等数字；
    且解析不出日期时返回空串，而不是伪造一个 20YY-01-01。
    """
    for t in reversed(texts):
        m = re.match(r'(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})', str(t).strip())
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ''


def parse_dlt_numbers(html):
    """
    解析500彩票网大乐透数据

    HTML格式：
      每行数据: 期号 前区5个号码 后区2个号码 ... 日期
      号码以空格分隔
    """
    if not html:
        return None

    # M6 修复: 删除原"方法1正则解析"。
    # 真实页面是 HTML 表格（</td><td> 分隔），纯文本行正则实测 0/68 行命中（死代码），
    # 且若页面出现 JS 内嵌的纯文本数字序列，会静默产出伪造记录（号码范围校验挡不住
    # 恰好合法的假数据）。只保留下方 BS4 表格解析作为唯一解析路径。

    records = []

    # 从HTML表格解析
    soup = BeautifulSoup(html, 'html.parser')
    # 找表格行
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 8:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        # 首列为期号
        period = texts[0]
        if not re.match(r'[23]\d{4}', period):
            continue
        # 提取号码（前5个数字在各自td中）
        try:
            front = []
            for idx in range(1, 6):
                n = int(re.search(r'\d+', texts[idx]).group())
                front.append(n)
            back = []
            for idx in range(6, 8):
                n = int(re.search(r'\d+', texts[idx]).group())
                back.append(n)
            if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                records.append({
                    'period': period,
                    'date': _pick_date(texts),
                    'front': sorted(front),
                    'back': sorted(back),
                })
        except (ValueError, AttributeError):
            continue

    # 去重
    seen = set()
    unique = []
    for r in records:
        if r['period'] not in seen:
            seen.add(r['period'])
            unique.append(r)

    return unique if unique else None


def scrape_dlt_range(start_period, end_period):
    """
    爬取指定期号范围的大乐透数据

    参数:
        start_period: 起始期号 (如 25138)
        end_period: 结束期号 (如 25160)

    返回:
        list of dict
    """
    print(f"  正在获取 {start_period}~{end_period} 期大乐透数据...")
    html = fetch_dlt_page(start_period, end_period)
    if not html:
        return None

    records = parse_dlt_numbers(html)
    if records:
        print(f"  成功解析 {len(records)} 期数据")
        # 按期号排序
        records.sort(key=lambda x: int(x['period']))
        return records

    return None


def fill_missing_data():
    """
    补全大乐透缺失数据

    策略：
      1. 检查数据库中最新一期
      2. 从最新一期+1开始爬取到最新期号
      3. 也检查同一年内的缺失
    """
    status = get_db_status()
    print(f"\n{'='*50}")
    print(f"  大乐透数据补全")
    print(f"{'='*50}")
    print(f"  当前: {status['count']} 条, 最新期 {status['max_period']} ({status['latest_date']})")
    if status['gaps']:
        print(f"  年内缺失 {len(status['gaps'])} 段: {status['gaps']}")
    else:
        print(f"  年内数据连续")

    with get_connection() as conn:
        total_added = 0

        latest = status['latest_period']

        # 空库场景：latest=0 时若从第1期爬，会落到 1~51 期（真实大乐透早已是 26xxx 期），
        # 什么都抓不到。改为从"去年年初"起爬（期号 YYNNN），靠下方翻页循环往后补齐，
        # 保证有足够历史期数供预测使用。
        if not latest:
            cur_yy = datetime.now().year % 100
            batch_start = (cur_yy - 1) * 1000 + 1  # 去年年初，如 2026 → 25001
        else:
            batch_start = latest + 1

        print(f"\n  正在检查 {batch_start} 期之后的新数据...")

        def _save_records(records):
            """将一批记录写入数据库，返回实际新增条数。

            多期修复：INSERT OR IGNORE 无法修复已存在的错误行，改与双色球一致的
            条件 upsert（仅当解析到合法日期才更新已有期号，自愈年内空洞）。
            H3：区间接口会回包含未开奖的预览期，日期晚于今天的一律跳过。
            """
            saved = 0
            today = datetime.now().date()
            for r in records:
                # 数据校验：确保记录合法性
                if not r.get('period') or not str(r['period']).strip().isdigit():
                    continue
                # H3 未来数据守卫：开奖日期晚于今天 → 未开奖预览期，跳过
                _d = r.get('date', '') or ''
                if _d:
                    try:
                        if datetime.strptime(_d, '%Y-%m-%d').date() > today:
                            log.warning("期%s 日期(%s)晚于今天，跳过未来数据", r['period'], _d)
                            continue
                    except ValueError:
                        pass
                front = r.get('front', [])
                if len(front) != 5 or not all(1 <= n <= 35 for n in front):
                    continue
                back = r.get('back', [])
                if len(back) != 2 or not all(1 <= n <= 12 for n in back):
                    continue
                try:
                    cur = conn.execute(
                        """INSERT INTO lottery_records
                           (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, back_1, back_2)
                           VALUES ('dlt', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(lottery_type, period) DO UPDATE SET
                             date=excluded.date,
                             front_1=excluded.front_1, front_2=excluded.front_2,
                             front_3=excluded.front_3, front_4=excluded.front_4,
                             front_5=excluded.front_5,
                             back_1=excluded.back_1, back_2=excluded.back_2
                           WHERE excluded.date <> ''""",
                        (r['period'], _d,
                         front[0], front[1], front[2], front[3], front[4],
                         back[0], back[1])
                    )
                    # rowcount 为本次语句实际影响行数，被 WHERE 跳过时为 0
                    if cur.rowcount > 0:
                        saved += 1
                except Exception as e:
                    log.warning("保存 %s 期失败: %s", r.get('period'), e)
            conn.commit()
            return saved

        # 先回补年内缺失的中段空洞：get_db_status 已算出 gaps（每段为
        # (前一期, 后一期, 缺失数)），逐段爬取区间中间的期号并入库，避免只在末尾追加新期。
        if status['gaps']:
            print(f"\n  正在回补 {len(status['gaps'])} 段年内缺失...")
            for prev_p, next_p, miss in status['gaps']:
                gap_records = scrape_dlt_range(prev_p + 1, next_p - 1)
                sleep_interval()
                if gap_records:
                    saved = _save_records(gap_records)
                    total_added += saved
                    print(f"  回补 {prev_p + 1}~{next_p - 1} 段: 新增 {saved} 期")

        # 主拉取：从 batch_start（最新期+1 或去年年初）起，按"年"整段请求
        # （start=YY001, end=YY999），替代逐页 +50 翻页——空库补 2 年历史从
        # ~24 次请求降到 2 次（与双色球按年区间做法一致）。按年 while 有上界
        # （最多当年），天然不会死循环；未开奖的预览期由 _save_records 未来守卫排除。
        cur_yy = datetime.now().year % 100
        pull_yy = batch_start // 1000
        while pull_yy <= cur_yy:
            block_start = batch_start if batch_start // 1000 == pull_yy else pull_yy * 1000 + 1
            block_end = pull_yy * 1000 + 999
            print(f"\n  拉取 {pull_yy} 年数据（{block_start}~{block_end}）...")
            records = scrape_dlt_range(block_start, block_end)
            sleep_interval()
            if not records:
                # M3 修复: 区分"正常终止"（当年已拉完/后续年份无数据）与"网络失败"。
                # 旧实现任一请求返回空就 break，导致今年数据完全不拉。
                # 当前年份(pull_yy == cur_yy)返回空多为已拉完——停止循环；
                # 更早年份返回空才可能是网络抖动，记录告警但继续尝试下一年。
                if pull_yy == cur_yy:
                    print(f"  {pull_yy} 年无更多数据，补全完成")
                else:
                    print(f"  ⚠️ {pull_yy} 年拉取为空（可能网络失败），继续下一年...")
                break
            saved = _save_records(records)
            total_added += saved
            print(f"  {pull_yy} 年: 解析 {len(records)} 期, 新增 {saved} 期 (至 {records[-1]['period']})")
            pull_yy += 1
            batch_start = pull_yy * 1000 + 1

    # 打印补全后状态
    new_status = get_db_status()
    print(f"\n{'='*50}")
    print(f"  补全完成!")
    print(f"  原来: {status['count']} 条")
    print(f"  现在: {new_status['count']} 条 (+{new_status['count'] - status['count']})")
    print(f"  最新: {new_status['max_period']} ({new_status['latest_date']})")
    print(f"{'='*50}")

    return new_status


if __name__ == '__main__':
    fill_missing_data()
