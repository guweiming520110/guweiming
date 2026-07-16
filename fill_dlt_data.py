#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透数据补全模块
从500彩票网爬取缺失的历史开奖数据
"""

import re
import time
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def _get_with_retry(url, max_retries=3, timeout=20):
    """带指数退避重试的 GET 请求辅助函数。

    对网络异常（超时/连接失败）与非 2xx 状态码做最多 max_retries 次重试，
    退避间隔为 1s、2s、4s……全部失败返回 None。
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            wait = 2 ** attempt
            print(f"  请求失败(第{attempt + 1}/{max_retries}次): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


def get_db_status():
    """检查数据库中大乐透数据的完整状态"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        "SELECT COUNT(*), MIN(period), MAX(period) FROM lottery_records WHERE lottery_type='dlt'"
    )
    cnt, pmin, pmax = cur.fetchone()

    cur = conn.execute(
        "SELECT period, date FROM lottery_records WHERE lottery_type='dlt' ORDER BY period"
    )
    rows = cur.fetchall()
    conn.close()

    periods = [int(r[0]) for r in rows]
    # 找缺失（同一年内）
    gaps = []
    for i in range(len(periods) - 1):
        gap = periods[i + 1] - periods[i]
        if gap > 1:
            # 检查是否为同年内缺失（期号为 YYNNN，前2位是年份）
            if str(periods[i])[:2] == str(periods[i + 1])[:2]:
                gaps.append((periods[i], periods[i + 1], gap - 1))

    return {
        'count': cnt,
        'min_period': pmin,
        'max_period': pmax,
        'latest_date': rows[-1][1] if rows else None,
        'gaps': gaps,
        'latest_period': int(pmax) if pmax else 0,
    }


def fetch_dlt_page(start_period, end_period):
    """从500彩票网获取大乐透数据页"""
    url = f'https://datachart.500.com/dlt/history/newinc/history.php?start={start_period}&end={end_period}'
    resp = _get_with_retry(url)
    if resp is None:
        return None
    resp.encoding = 'gb2312'
    return resp.text


def parse_dlt_numbers(html):
    """
    解析500彩票网大乐透数据

    HTML格式：
      每行数据: 期号 前区5个号码 后区2个号码 ... 日期
      号码以空格分隔
    """
    if not html:
        return None

    records = []

    # 方法1: 直接用正则匹配数字序列
    # 格式如: 2515013141528310105942,601,2234...
    # 关键: 期号(5位) + 前区5个号码(各1-2位) + 后区2个号码(各1-2位) + 其他数据

    # 查找包含期号和开奖号码的行
    lines = html.split('\n')
    for line in lines:
        line = line.strip()
        # 匹配期号 (5位数字)
        period_match = re.search(r'([23]\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})', line)
        if period_match:
            period = period_match.group(1)
            front_nums = [int(period_match.group(i)) for i in range(2, 7)]
            back_nums = [int(period_match.group(i)) for i in range(7, 9)]
            if all(1 <= n <= 35 for n in front_nums) and all(1 <= n <= 12 for n in back_nums):
                # 提取日期 - 找行中的 YYYY-MM-DD 或 YYYY年MM月DD日
                date_match = re.search(r'(\d{4})[-年](\d{1,2})[-月](\d{1,2})', line)
                if date_match:
                    date_str = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
                else:
                    # 从期号推算大致年份（期号为 YYNNN，前2位是年份后两位）
                    year = f"20{period[:2]}"
                    date_str = f"{year}-01-01"

                records.append({
                    'period': period,
                    'date': date_str,
                    'front': sorted(front_nums),
                    'back': sorted(back_nums),
                })

    if records:
        # 去重
        seen = set()
        unique = []
        for r in records:
            if r['period'] not in seen:
                seen.add(r['period'])
                unique.append(r)
        return unique

    # 方法2: 从HTML表格解析
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
                    'date': texts[-1] if len(texts) > 8 else f"20{period[:2]}-01-01",
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

    conn = sqlite3.connect(str(DB_PATH))
    total_added = 0

    latest = status['latest_period']

    # 空库场景：latest=0 时若从第1期爬，会落到 1~51 期（真实大乐透早已是 26xxx 期），
    # 什么都抓不到。改为从"去年年初"起爬（期号 YYNNN），靠下方 while 循环往后补齐，
    # 保证有足够历史期数供预测使用。
    if not latest:
        cur_yy = datetime.now().year % 100
        batch_start = (cur_yy - 1) * 1000 + 1  # 去年年初，如 2026 → 25001
    else:
        batch_start = latest + 1
    batch_end = batch_start + 50

    print(f"\n  正在检查 {batch_start} 期之后的新数据...")

    def _save_records(records):
        """将一批记录写入数据库，返回实际保存条数"""
        saved = 0
        for r in records:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO lottery_records
                       (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, back_1, back_2)
                       VALUES ('dlt', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r['period'], r['date'],
                     r['front'][0], r['front'][1], r['front'][2], r['front'][3], r['front'][4],
                     r['back'][0], r['back'][1])
                )
                saved += 1
            except Exception:
                pass
        conn.commit()
        return saved

    # 先回补年内缺失的中段空洞：get_db_status 已算出 gaps（每段为
    # (前一期, 后一期, 缺失数)），逐段爬取区间中间的期号并入库，避免只在末尾追加新期。
    if status['gaps']:
        print(f"\n  正在回补 {len(status['gaps'])} 段年内缺失...")
        for prev_p, next_p, miss in status['gaps']:
            gap_records = scrape_dlt_range(prev_p + 1, next_p - 1)
            if gap_records:
                saved = _save_records(gap_records)
                total_added += saved
                print(f"  回补 {prev_p + 1}~{next_p - 1} 段: 新增 {saved} 期")

    records = scrape_dlt_range(batch_start, batch_end)
    if records:
        saved = _save_records(records)
        total_added += saved
        print(f"  新增 {saved} 期数据")

        # 如果这次获取到了数据，继续获取后续
        while len(records) >= 50:
            batch_start = int(records[-1]['period']) + 1
            batch_end = batch_start + 50
            records = scrape_dlt_range(batch_start, batch_end)
            if not records:
                break
            saved = _save_records(records)
            total_added += saved
            print(f"  追加 {saved} 期 (至 {records[-1]['period']})")

    # 跨年续爬：大乐透期号按年重置（如 ...25150 → 26001），区间接口从 25151 起会返回空，
    # 导致今年数据补不上。补完去年后，若最新期仍停在去年，则从今年年初继续爬一轮。
    cur_yy = datetime.now().year % 100
    last_period = int(conn.execute(
        "SELECT MAX(CAST(period AS INTEGER)) FROM lottery_records WHERE lottery_type='dlt'"
    ).fetchone()[0] or 0)
    if last_period and (last_period // 1000) < cur_yy:
        batch_start = cur_yy * 1000 + 1  # 今年年初，如 26001
        batch_end = batch_start + 50
        print(f"\n  跨年续爬 {cur_yy} 年数据（从 {batch_start} 起）...")
        records = scrape_dlt_range(batch_start, batch_end)
        while records:
            saved = _save_records(records)
            total_added += saved
            print(f"  追加 {saved} 期 (至 {records[-1]['period']})")
            if len(records) < 50:
                break
            batch_start = int(records[-1]['period']) + 1
            batch_end = batch_start + 50
            records = scrape_dlt_range(batch_start, batch_end)

    conn.close()

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
