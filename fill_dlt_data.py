#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透数据补全模块
从500彩票网爬取缺失的历史开奖数据
"""

import re
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
            # 检查是否为跨年
            if str(periods[i])[:3] == str(periods[i + 1])[:3]:
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
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = 'gb2312'
        return resp.text
    except Exception as e:
        print(f"  请求失败: {e}")
        return None


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
        period_match = re.search(r'(2\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})', line)
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
                    # 从期号推算大致年份
                    year = period[:4]
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
        if not re.match(r'2\d{4}', period):
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
                    'date': texts[-1] if len(texts) > 8 else f"{period[:4]}-01-01",
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

    # 确定要获取的范围：从最新期+1到当前最新
    # 先试探最新期号
    test_periods = []
    year = int(str(latest)[:4])
    for y in [year, year + 1]:
        for p in range(1, 160, 10):
            test_periods.append(int(f"{y}{p:03d}"))

    # 从最新期开始，分批获取后续数据
    batch_start = latest + 1
    batch_end = batch_start + 50

    print(f"\n  正在检查 {batch_start} 期之后的新数据...")

    records = scrape_dlt_range(batch_start, batch_end)
    if records:
        saved = 0
        for r in records:
            try:
                front_str = ','.join(str(n) for n in r['front'])
                back_str = ','.join(str(n) for n in r['back'])
                conn.execute(
                    """INSERT OR REPLACE INTO lottery_records 
                       (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, back_1, back_2)
                       VALUES ('dlt', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r['period'], r['date'],
                     r['front'][0], r['front'][1], r['front'][2], r['front'][3], r['front'][4],
                     r['back'][0], r['back'][1])
                )
                saved += 1
            except Exception as e:
                pass
        conn.commit()
        total_added += saved
        print(f"  新增 {saved} 期数据")

        # 如果这次获取到了数据，继续获取后续
        while len(records) >= 50:
            batch_start = int(records[-1]['period']) + 1
            batch_end = batch_start + 50
            records = scrape_dlt_range(batch_start, batch_end)
            if not records:
                break
            saved = 0
            for r in records:
                try:
                    front_str = ','.join(str(n) for n in r['front'])
                    back_str = ','.join(str(n) for n in r['back'])
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
            total_added += saved
            print(f"  追加 {saved} 期 (至 {records[-1]['period']})")

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
