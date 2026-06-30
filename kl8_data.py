#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快乐8 真实数据采集模块
从500彩票网(https://datachart.500.com/kl8/) 获取历史开奖数据
"""

import re
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'

KL8_NUMS = 80
KL8_DRAWN = 20

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def create_kl8_table(conn=None):
    """创建快乐8数据表"""
    should_close = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kl8_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL UNIQUE,
            date TEXT NOT NULL DEFAULT '',
            numbers TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kl8_period ON kl8_records(period)")
    conn.commit()
    if should_close:
        conn.close()


def fetch_kl8_html(num_periods=50):
    """
    从500彩票网获取快乐8走势图HTML

    参数:
        num_periods: 30 / 50 / 100
    """
    url = f'https://datachart.500.com/kl8?expect={num_periods}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = 'gb2312'
        return resp.text
    except Exception as e:
        print(f"  请求失败: {e}")
        return None


def parse_kl8_numbers(html):
    """
    从500彩票网走势图HTML中解析出每期的开奖号码

    HTML结构:
      - 数据在 <tbody id="tdata"> 内
      - 每行 <tr> 对应一期，首列为期号
      - 后续80列为号码1-80
      - 开奖号码的td带有 class="chartBall01"
      - 列索引（从1开始）就是号码本身

    返回:
        list of dict: [{'period':'2026157', 'numbers':[1,3,5,...]}, ...]
    """
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # 优先查找 tbody id="tdata" (2026年网站结构)
    tbody = soup.find('tbody', id='tdata')
    if tbody:
        records = _parse_from_tbody(tbody)
        if records:
            return records

    # 备用方法：查找chartTbl表格
    table = soup.find('table', class_=re.compile(r'chartTbl', re.I))
    if table:
        records = _parse_from_table(table)
        if records:
            return records

    # 最后备用：直接从HTML文本提取
    records = _parse_from_text(html)
    if records:
        return records

    return None


def _parse_from_tbody(tbody):
    """从 tbody id="tdata" 解析（2026年网站结构）"""
    records = []
    rows = tbody.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 5:
            continue

        # 首列期号
        period_td = cells[0].get_text(strip=True)
        period_match = re.match(r'(\d{7})', period_td)
        if not period_match:
            continue
        period = period_match.group(1)

        # 关键修正: 列索引（从1开始）就是号码本身
        # class="chartBall01" 的td表示开奖号码
        numbers = []
        for idx, td in enumerate(cells[1:], 1):  # idx = 1~80 就是号码
            if idx > KL8_NUMS:
                break
            # 检查是否是开奖号码
            td_class = td.get('class', [])
            if 'chartBall01' in td_class or 'chartBall02' in td_class:
                numbers.append(idx)

        if len(numbers) >= KL8_DRAWN:
            records.append({
                'period': period,
                'numbers': sorted(numbers[:KL8_DRAWN])
            })

    return records if records else None


def _parse_from_text(html):
    """
    直接从HTML文本提取开奖号码
    查找 class="chartBall01" 的td标签，列索引就是号码
    """
    records = []

    # 查找数据行: <tr> 包含期号 <td>
    row_pattern = r'<tr[^>]*>(.*?)</tr>'
    rows = re.findall(row_pattern, html, re.DOTALL)

    for row_html in rows:
        # 提取期号
        period_match = re.search(r'<td[^>]*>\s*(\d{7})\s*</td>', row_html)
        if not period_match:
            continue
        period = period_match.group(1)

        # 提取所有td，找到包含 chartBall01 的td，它的位置就是号码
        td_pattern = r'<td[^>]*class="[^"]*(chartBall01|chartBall02)[^"]*"[^>]*>.*?</td>'
        td_matches = list(re.finditer(td_pattern, row_html))

        # 找出这些td在该行中的位置索引（从1开始）
        all_td_pattern = r'<td[^>]*>.*?</td>'
        all_tds = list(re.finditer(all_td_pattern, row_html))

        numbers = []
        for ball_match in td_matches:
            # 找到这个td在所有td中的位置
            for idx, td_match in enumerate(all_tds[1:], 1):  # 跳过期号td
                if td_match.start() == ball_match.start():
                    numbers.append(idx)
                    break

        if len(numbers) >= KL8_DRAWN:
            records.append({
                'period': period,
                'numbers': sorted(numbers[:KL8_DRAWN])
            })

    return records if records else None


def _parse_from_table(table):
    """从BeautifulSoup Table对象解析"""
    records = []
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 5:
            continue
        # 首列期号
        period_td = cells[0].get_text(strip=True)
        period_match = re.match(r'(\d{7})', period_td)
        if not period_match:
            continue
        period = period_match.group(1)

        numbers = []
        for idx, td in enumerate(cells[1:], 1):  # idx = 1~80 就是号码
            if idx > KL8_NUMS:
                break
            # 检查是否是开奖号码
            td_class = td.get('class', [])
            if 'chartBall01' in td_class or 'chartBall02' in td_class:
                numbers.append(idx)

        if len(numbers) >= KL8_DRAWN:
            records.append({
                'period': period,
                'numbers': sorted(numbers[:KL8_DRAWN])
            })

    return records if records else None


def scrape_from_500(num_periods=50):
    """
    从500彩票网采集快乐8真实历史数据

    返回:
        list of dict or None
    """
    print(f"  正在从500.com获取{num_periods}期快乐8数据...")
    html = fetch_kl8_html(num_periods)
    if not html:
        return None

    records = parse_kl8_numbers(html)
    if records:
        print(f"  成功解析 {len(records)} 期真实开奖数据")
        for r in records[:3]:
            print(f"    期号 {r['period']}: {r['numbers'][:10]}...")
        return records

    return None


def save_to_db(records):
    """保存到数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    create_kl8_table(conn)

    saved = 0
    for r in records:
        try:
            numbers_str = ','.join(str(n) for n in r['numbers'])
            cur = conn.execute(
                "INSERT OR IGNORE INTO kl8_records (period, date, numbers) VALUES (?, ?, ?)",
                (r['period'], r.get('date', ''), numbers_str)
            )
            # rowcount 为本条语句实际插入行数（被 IGNORE 跳过则为0），
            # 不能用连接级累计的 total_changes 判断
            if cur.rowcount > 0:
                saved += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return saved


def load_kl8_data(limit=None):
    """从数据库加载快乐8数据"""
    conn = sqlite3.connect(str(DB_PATH))
    create_kl8_table(conn)

    try:
        df = pd.read_sql("SELECT * FROM kl8_records ORDER BY period ASC", conn)
        conn.close()

        if len(df) == 0:
            return None

        all_nums = []
        for nums_str in df['numbers']:
            nums = [int(x) for x in str(nums_str).split(',') if x.strip().isdigit()]
            all_nums.append(nums)
        df['num_list'] = all_nums

        if limit and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  加载失败: {e}")
        conn.close()
        return None


def ensure_kl8_data(num_periods=50, force_refresh=False):
    """
    确保快乐8真实数据可用（仅从500.com获取，不使用模拟数据）
    """
    # 先检查数据库
    df = load_kl8_data(limit=num_periods)
    if df is not None and len(df) >= min(num_periods, 20) and not force_refresh:
        print(f"  从数据库加载 {len(df)} 期快乐8数据")
        return df

    # 从500.com爬取真实数据
    records = scrape_from_500(num_periods)

    if not records:
        print("  ❌ 无法从500.com获取数据，请检查网络连接")
        print("  提示: 可尝试手动访问 https://datachart.500.com/kl8/ 确认网站是否可访问")
        return None

    # 保存到数据库
    saved = save_to_db(records)
    print(f"  已保存 {saved} 期新数据到数据库")

    df = load_kl8_data(limit=num_periods)
    if df is not None and len(df) > 0:
        return df

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='快乐8数据采集')
    parser.add_argument('--periods', type=int, default=50, help='期数')
    parser.add_argument('--force', action='store_true', help='强制刷新')
    args = parser.parse_args()

    df = ensure_kl8_data(num_periods=args.periods, force_refresh=args.force)
    if df is not None:
        print(f"\n共 {len(df)} 期快乐8真实数据")
        for _, row in df.head(3).iterrows():
            nums = row['num_list']
            print(f"  期号 {row['period']}: {nums[:10]}... ({len(nums)}个号码)")
    else:
        print("\n❌ 未获取到数据")
