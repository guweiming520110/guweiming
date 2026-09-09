#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球数据补全模块
从500彩票网爬取历史开奖数据
"""

import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup

from config import HTTP_HEADERS, HTTP_ENCODING, HTTP_MAX_RETRIES, HTTP_TIMEOUT, HTTP_REQUEST_INTERVAL
from db_utils import get_connection
from http_utils import get_with_retry, sleep_interval
from logger import log

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

HEADERS = HTTP_HEADERS


def _get_with_retry(url, max_retries=HTTP_MAX_RETRIES, timeout=HTTP_TIMEOUT):
    """向后兼容包装：统一委托 http_utils.get_with_retry（指数退避重试）"""
    return get_with_retry(url, max_retries=max_retries, timeout=timeout,
                          encoding=HTTP_ENCODING)


def get_ssq_status():
    """检查数据库双色球数据状态"""
    with get_connection(commit=False) as conn:
        cur = conn.execute(
            "SELECT COUNT(*), MIN(period), MAX(period) FROM lottery_records WHERE lottery_type='ssq'"
        )
        cnt, pmin, pmax = cur.fetchone()
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
    # 编码已由 http_utils 统一设置（GBK），此处不再硬编码 gb2312
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

    with get_connection() as conn:

        def _save_records(records):
            """将一批记录写入数据库，返回实际新增条数。

            用 ON CONFLICT 原子 upsert 替代 INSERT OR REPLACE：
            REPLACE 语义是"先删后插"，会用空日期覆盖历史正确数据并重置 id；
            这里改为仅当本次解析到合法日期时才更新已有期号。
            """
            saved = 0
            for r in records:
                # 数据校验：确保记录合法性
                if not r.get('period') or not str(r['period']).strip().isdigit():
                    continue
                front = r.get('front', [])
                if len(front) != 6 or not all(1 <= n <= 33 for n in front):
                    continue
                back = r.get('back', [])
                if isinstance(back, int):
                    back = [back]
                if len(back) != 1 or not all(1 <= n <= 16 for n in back):
                    continue
                try:
                    cur = conn.execute(
                        """INSERT INTO lottery_records
                           (lottery_type, period, date, front_1, front_2, front_3,
                            front_4, front_5, front_6, back_1)
                           VALUES ('ssq', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(lottery_type, period) DO UPDATE SET
                             date=excluded.date,
                             front_1=excluded.front_1, front_2=excluded.front_2,
                             front_3=excluded.front_3, front_4=excluded.front_4,
                             front_5=excluded.front_5, front_6=excluded.front_6,
                             back_1=excluded.back_1
                           WHERE excluded.date <> ''""",
                        (r['period'], r.get('date', ''),
                         front[0], front[1], front[2], front[3], front[4], front[5],
                         back[0])
                    )
                    # rowcount 为本次语句实际影响行数，被 WHERE 跳过时为 0
                    if cur.rowcount > 0:
                        saved += 1
                except Exception as e:
                    log.warning("保存 %s 期失败: %s", r.get('period'), e)
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
            sleep_interval()
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

        return new_status


if __name__ == '__main__':
    fill_ssq_data()
