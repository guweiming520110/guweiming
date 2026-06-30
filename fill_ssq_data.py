#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球数据补全模块
从500彩票网爬取历史开奖数据
"""

import re
import sqlite3
import pandas as pd
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


def get_ssq_status():
    """检查数据库双色球数据状态"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        "SELECT COUNT(*), MIN(period), MAX(period) FROM lottery_records WHERE lottery_type='ssq'"
    )
    cnt, pmin, pmax = cur.fetchone()
    conn.close()
    return {'count': cnt, 'min_period': pmin, 'max_period': pmax}


def fetch_ssq_history_page(start=1, end=100):
    """从500彩票网获取双色球历史数据页面

    注意：500.com 双色球 history.shtml 页面不支持 start/end 区间参数，
    实际仅返回默认页（约最近100期）。start/end 暂保留以兼容调用方，但不生效。
    如需按区间补全历史，需改用支持区间的接口。
    """
    url = f'https://datachart.500.com/ssq/history/history.shtml'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = 'gb2312'
        return resp.text
    except Exception as e:
        print(f"  请求失败: {e}")
        return None


def parse_ssq_numbers(html):
    """
    从500彩票网历史数据HTML解析双色球开奖号码

    HTML表格格式:
      期号 | 红球1 | 红球2 | ... | 红球6 | 蓝球 | ... | 开奖日期
    """
    if not html:
        return None

    records = []
    soup = BeautifulSoup(html, 'html.parser')

    # 查找数据表格（历史数据页面）
    # 在表格中找到包含期号的行
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) < 8:
                continue

            # 期号：5-7位数字
            period_match = re.match(r'(\d{5,7})', texts[0])
            if not period_match:
                continue
            period = period_match.group(1)

            # 提取6个红球 (1-33)
            try:
                front = []
                for idx in range(1, 7):
                    val = int(re.search(r'\d+', texts[idx]).group())
                    front.append(val)
                # 提取蓝球
                back = int(re.search(r'\d+', texts[7]).group())
            except (ValueError, AttributeError, IndexError):
                continue

            if not (all(1 <= n <= 33 for n in front) and 1 <= back <= 16):
                continue

            # 提取开奖日期（最后一列或倒数第二列）
            date_str = ''
            for t in reversed(texts):
                if re.match(r'\d{4}-\d{2}-\d{2}', t):
                    date_str = t
                    break

            records.append({
                'period': period,
                'date': date_str,
                'front': sorted(front),
                'back': back,
            })

    if records:
        # 去重
        seen = set()
        unique = []
        for r in records:
            if r['period'] not in seen:
                seen.add(r['period'])
                unique.append(r)
        unique.sort(key=lambda x: int(x['period']))
        return unique

    return None


def fill_ssq_data():
    """补全双色球数据"""
    status = get_ssq_status()
    print(f"\n{'='*50}")
    print(f"  双色球数据补全")
    print(f"{'='*50}")
    print(f"  当前: {status['count']} 条, 最新期 {status['max_period']}")

    conn = sqlite3.connect(str(DB_PATH))

    # 直接从500.com获取最新100期数据
    print(f"\n  正在从500.com获取双色球数据...")
    html = fetch_ssq_history_page()
    if not html:
        print("  ❌ 获取数据失败")
        conn.close()
        return

    records = parse_ssq_numbers(html)
    if not records:
        print("  ❌ 解析数据失败")
        conn.close()
        return

    print(f"  成功解析 {len(records)} 期数据")
    print(f"  范围: {records[0]['period']} ~ {records[-1]['period']}")

    # 只保存数据库中不存在的期号
    existing = set(r[0] for r in conn.execute(
        "SELECT period FROM lottery_records WHERE lottery_type='ssq'"
    ).fetchall())

    new_records = [r for r in records if r['period'] not in existing]
    print(f"  新增: {len(new_records)} 期")

    saved = 0
    for r in new_records:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO lottery_records 
                   (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, front_6, back_1)
                   VALUES ('ssq', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r['period'], r['date'],
                 r['front'][0], r['front'][1], r['front'][2],
                 r['front'][3], r['front'][4], r['front'][5],
                 r['back'])
            )
            saved += 1
        except Exception as e:
            pass
    conn.commit()

    # 打印补全后状态
    new_status = get_ssq_status()
    print(f"\n{'='*50}")
    print(f"  补全完成!")
    print(f"  原来: {status['count']} 条")
    print(f"  现在: {new_status['count']} 条 (+{new_status['count'] - status['count']})")
    print(f"  最新: {new_status['max_period']}")
    print(f"{'='*50}")

    conn.close()
    return new_status


if __name__ == '__main__':
    fill_ssq_data()
