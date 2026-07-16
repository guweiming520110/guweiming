#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球数据补全模块
从500彩票网爬取历史开奖数据
"""

import re
import time
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
    """从500彩票网按区间获取双色球历史数据页面。

    使用 datachart.500.com 的 newinc/history.php 区间接口（与大乐透同源），
    支持 start/end 期号区间参数，可按年批量补全历史。
    期号格式 YYNNN（如 25001 = 2025年第1期）。

    历史教训：旧实现用 history.shtml 页面且忽略区间参数，只能拿到最近一页
    （且解析后仅入库 30 余期），导致双色球历史数据长期不足、无法训练模型。
    """
    url = (
        f'https://datachart.500.com/ssq/history/newinc/history.php'
        f'?start={start}&end={end}'
    )
    resp = _get_with_retry(url)
    if resp is None:
        return None
    resp.encoding = 'gb2312'
    return resp.text


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
    """补全双色球数据。

    采用区间接口按年分批爬取（与大乐透一致）：
      - 空库场景：从去年年初（YYNNN，如 25001）起爬，保证有足够历史供训练
      - 已有数据：从最新期之后继续补
      - 跨年续爬：双色球期号按年重置（如 25151 → 26001），补完去年后若最新期
        仍停在去年，则从今年年初再爬一轮
    """
    status = get_ssq_status()
    print(f"\n{'='*50}")
    print(f"  双色球数据补全")
    print(f"{'='*50}")
    print(f"  当前: {status['count']} 条, 最新期 {status['max_period']}")

    conn = sqlite3.connect(str(DB_PATH))

    def _save_records(records):
        """将一批记录写入数据库，只插入不存在的期号，返回实际新增条数"""
        existing = set(r[0] for r in conn.execute(
            "SELECT period FROM lottery_records WHERE lottery_type='ssq'"
        ).fetchall())
        saved = 0
        for r in records:
            if r['period'] in existing:
                continue
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
            except Exception:
                pass
        conn.commit()
        return saved

    def _scrape_year(start_yy):
        """爬取某年（YY）全年数据并入库，返回该年实际新增条数。

        双色球每年约 150 期，一次区间请求（start=YY001, end=YY999）即可覆盖全年。
        """
        yy = start_yy
        start_p = yy * 1000 + 1        # 如 25001
        end_p = yy * 1000 + 999        # 如 25999
        html = fetch_ssq_history_page(start=start_p, end=end_p)
        if not html:
            print(f"  ⚠️ {yy} 年数据请求失败")
            return 0
        records = parse_ssq_numbers(html)
        if not records:
            return 0
        saved = _save_records(records)
        print(f"  {yy} 年: 解析 {len(records)} 期, 新增 {saved} 期 "
              f"(范围 {records[0]['period']}~{records[-1]['period']})")
        return saved

    total_added = 0
    cur_yy = datetime.now().year % 100

    print(f"\n  正在从500.com按年区间补全双色球数据...")
    if not status['count']:
        # 空库：先补去年再补今年，保证历史充足
        total_added += _scrape_year(cur_yy - 1)
    total_added += _scrape_year(cur_yy)

    # 打印补全后状态
    new_status = get_ssq_status()
    print(f"\n{'='*50}")
    print(f"  补全完成!")
    print(f"  原来: {status['count']} 条")
    print(f"  现在: {new_status['count']} 条 (+{total_added})")
    print(f"  最新: {new_status['max_period']}")
    print(f"{'='*50}")

    conn.close()
    return new_status


if __name__ == '__main__':
    fill_ssq_data()
