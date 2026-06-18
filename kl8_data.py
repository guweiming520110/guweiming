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
      - 数据在 <div id="container"> 内的 chartTbl 表格中
      - 每行 <tr> 对应一期，首列为期号
      - 后续80列为号码1-80
      - 开奖号码以红/蓝色圆球图片或彩色数字显示
      - 遗漏值以灰色数字显示

    返回:
        list of dict: [{'period':'2026157', 'numbers':[1,3,5,...]}, ...]
    """
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # 定位数据表格
    # 方法1: 查找id为container的div
    container = soup.find('div', id='container')
    if not container:
        # 方法2: 查找chartTbl表格
        table = soup.find('table', class_=re.compile(r'chartTbl', re.I))
        if not table:
            # 方法3: 直接找包含期号数字的表格
            tables = soup.find_all('table')
            for t in tables:
                if t.get('width') == '100%' and not t.get('class'):
                    continue
                if t.find('td', string=re.compile(r'\d{7}')):
                    table = t
                    break

    # 如果以上都没找到，直接从HTML文本中提取数据
    records = _parse_from_text(html)
    if records:
        return records

    # 尝试从table解析
    if table:
        records = _parse_from_table(table)
        if records:
            return records

    return None


def _parse_from_text(html):
    """
    直接从HTML文本提取开奖号码
    从markdown输出中我们看到数据是以表格形式呈现的，
    每行的期号后跟着80个数字，其中值等于列号的代表开奖号码
    """
    records = []

    # 查找数据行: <tr> 包含期号 <td> 和80个号码 <td>
    # 通过匹配 <tr> 内的内容来提取
    row_pattern = r'<tr[^>]*>(.*?)</tr>'
    td_pattern = r'<td[^>]*>(.*?)</td>'

    rows = re.findall(row_pattern, html, re.DOTALL)
    for row_html in rows:
        # 提取所有td
        cells = re.findall(td_pattern, row_html, re.DOTALL)
        if len(cells) < 5:
            continue

        # 清理HTML标签
        clean_cells = []
        for c in cells:
            # 取纯文本
            text = re.sub(r'<[^>]+>', '', c).strip()
            clean_cells.append(text)

        # 第一列应该是期号（7位数字）
        period_match = re.match(r'(\d{7})', clean_cells[0])
        if not period_match:
            continue
        period = period_match.group(1)

        # 后续80列是号码数据
        # 如果一个格子的值等于列索引，则该号码被开出
        # 注意：数据列可能有遗漏值标记，开奖号码会以彩色显示
        numbers = []
        for idx in range(1, min(len(clean_cells), KL8_NUMS + 1)):
            val = clean_cells[idx].strip()
            if not val:
                continue
            try:
                n = int(val)
            except ValueError:
                continue
            if n == idx:  # 值等于列号 = 开奖号码
                numbers.append(n)

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
        for idx, td in enumerate(cells[1:], 1):
            if idx > KL8_NUMS:
                break
            text = td.get_text(strip=True)
            if not text:
                continue
            try:
                n = int(text)
            except ValueError:
                continue
            # 检查是否有红/蓝样式（开奖号码通常有特殊样式）
            has_ball = (
                td.find('font', attrs={'color': re.compile(r'red|blue|ff|#', re.I)})
                or td.find('span', class_=re.compile(r'red|blue|ball', re.I))
                or td.find('img', src=re.compile(r'red|blue'))
            )
            if n == idx:
                numbers.append(n)

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
            conn.execute(
                "INSERT OR IGNORE INTO kl8_records (period, date, numbers) VALUES (?, ?, ?)",
                (r['period'], r.get('date', ''), numbers_str)
            )
            if conn.total_changes > 0:
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
