#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快乐8 真实数据采集模块
从500彩票网(https://datachart.500.com/kl8/) 获取历史开奖数据
"""

import re
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from bs4 import BeautifulSoup

from config import DB_PATH, HTTP_HEADERS, HTTP_ENCODING, HTTP_MAX_RETRIES, HTTP_TIMEOUT
from db_utils import get_connection
from http_utils import get_with_retry
from logger import log
from schema import DDL_KL8_RECORDS

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

KL8_NUMS = 80
KL8_DRAWN = 20
# 500.com 走势图接口仅支持这几档期数，避免传入任意值构造出无效 URL
VALID_PERIODS = (30, 50, 100)

HEADERS = HTTP_HEADERS


def period_to_date(period):
    """期号 YYYYNNN → 开奖日期 (快乐8每天开一期，第1期=1月1日，闰年自动处理)。

    500.com 走势图无日期列导致 date 字段全空(C4修复)，
    官方规则确定为每日1期，可由期号精确反推；无法反推时返回 ''。
    """
    period = str(period).strip()
    if len(period) < 5 or not period.isdigit():
        return ''
    year = int(period[:4])
    seq = int(period[4:])
    try:
        base = datetime(year, 1, 1)
        return (base + timedelta(days=seq - 1)).strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return ''


def _get_with_retry(url, max_retries=HTTP_MAX_RETRIES, timeout=HTTP_TIMEOUT):
    """向后兼容包装：统一委托 http_utils.get_with_retry（指数退避重试）"""
    return get_with_retry(url, max_retries=max_retries, timeout=timeout,
                          encoding=HTTP_ENCODING)


def create_kl8_table(conn=None):
    """创建快乐8数据表（DDL 统一取自 schema，避免两份定义漂移）"""
    if conn is None:
        with get_connection() as _conn:
            _conn.execute(DDL_KL8_RECORDS)
        return
    conn.execute(DDL_KL8_RECORDS)
    conn.commit()


def fetch_kl8_html(num_periods=50):
    """
    从500彩票网获取快乐8走势图HTML

    参数:
        num_periods: 30 / 50 / 100（其他值自动向上取到最近的合法档位）
    """
    if num_periods not in VALID_PERIODS:
        candidates = [p for p in VALID_PERIODS if p >= num_periods]
        num_periods = candidates[0] if candidates else max(VALID_PERIODS)
    url = f'https://datachart.500.com/kl8?expect={num_periods}'
    resp = get_with_retry(url, encoding=HTTP_ENCODING)
    if resp is None:
        return None
    return resp.text


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


def _extract_date(cells):
    """从td单元格列表中提取日期字符串（YYYY-MM-DD格式），找不到返回空字符串"""
    for td in cells:
        text = td.get_text(strip=True)
        m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if m:
            return m.group(1)
    return ''


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
            # H1: 仅 chartBall01 是真实开奖号码；chartBall02 语义未确认为非中奖态，
            # 并入会产生每期多出几十个号码 → 截断后错误入库污染训练
            td_class = td.get('class', [])
            if 'chartBall01' in td_class:
                numbers.append(idx)

        # H2: 数量必须恰好等于 20，否则整行拒收——畸形行会谎报为正常数据
        if len(numbers) != KL8_DRAWN:
            log.warning("期号 %s 解析出 %d 个号码(应为 %d)，整行丢弃",
                        period, len(numbers), KL8_DRAWN)
            continue
        # 尝试从行末列提取日期（YYYY-MM-DD 格式）
        date_str = _extract_date(cells)
        records.append({
            'period': period,
            'date': date_str,
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
        # H1: 只认 chartBall01（真实开奖球），chartBall02 语义未确认为非中奖态
        td_pattern = r'<td[^>]*class="[^"]*chartBall01[^"]*"[^>]*>.*?</td>'
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

        # H2: 数量必须恰好 20，畸形行整行拒收
        if len(numbers) != KL8_DRAWN:
            log.warning("期号 %s 解析出 %d 个号码(应为 %d)，整行丢弃",
                        period, len(numbers), KL8_DRAWN)
            continue
        # 该分支只有 HTML 文本、没有 td 标签对象，因此改用正则提取日期
        _m = re.search(r'(\d{4}-\d{2}-\d{2})', row_html)
        date_str = _m.group(1) if _m else ''
        records.append({
            'period': period,
            'date': date_str,
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
            # H1: 仅 chartBall01 是真实开奖号码；chartBall02 语义未确认为非中奖态，
            # 并入会产生每期多出几十个号码 → 截断后错误入库污染训练
            td_class = td.get('class', [])
            if 'chartBall01' in td_class:
                numbers.append(idx)

        # H2: 数量必须恰好等于 20，否则整行拒收——畸形行会谎报为正常数据
        if len(numbers) != KL8_DRAWN:
            log.warning("期号 %s 解析出 %d 个号码(应为 %d)，整行丢弃",
                        period, len(numbers), KL8_DRAWN)
            continue
        # 尝试从行末列提取日期（YYYY-MM-DD 格式）
        date_str = _extract_date(cells)
        records.append({
            'period': period,
            'date': date_str,
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
            date_info = f" ({r.get('date', '')})" if r.get('date') else ''
            print(f"    期号 {r['period']}{date_info}: {r['numbers'][:10]}...")
        return records

    return None


def save_to_db(records):
    """保存到数据库

    C4: date 缺失时按期号反推补全（500.com 走势图无日期列）；
    H3: 反推日期晚于今天 ＝ 站点回包未来数据，整条跳过，绝不污染历史表。
    """
    saved = 0
    today = datetime.now().date()
    with get_connection() as conn:
        conn.execute(DDL_KL8_RECORDS)
        for r in records:
            try:
                numbers_str = ','.join(str(n) for n in r['numbers'])
                date_str = (r.get('date') or period_to_date(r['period'])) or ''
                # 未来数据守卫：期号反推出的日期晚于今天，说明读到了未开奖的预览期
                if date_str:
                    try:
                        d = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if d > today:
                            log.warning("期%s 日期(%s)晚于今天，跳过未来数据", r['period'], date_str)
                            continue
                    except ValueError:
                        pass
                cur = conn.execute(
                    "INSERT OR IGNORE INTO kl8_records (period, date, numbers) VALUES (?, ?, ?)",
                    (r['period'], date_str, numbers_str)
                )
                # rowcount 为本条语句实际插入行数（被 IGNORE 跳过则为0），
                # 不能用连接级累计的 total_changes 判断
                if cur.rowcount > 0:
                    saved += 1
            except Exception as e:
                log.warning("保存 %s 期失败: %s", r.get('period'), e)
    return saved


def load_kl8_data(limit=None):
    """从数据库加载快乐8数据（limit 在 SQL 层生效，避免全表读入内存）"""
    try:
        with get_connection(commit=False) as conn:
            conn.execute(DDL_KL8_RECORDS)
            if limit:
                df = pd.read_sql(
                    "SELECT period, date, numbers FROM kl8_records "
                    "ORDER BY period DESC LIMIT ?", conn, params=(int(limit),)
                )
                df = df.iloc[::-1].reset_index(drop=True)
            else:
                df = pd.read_sql(
                    "SELECT period, date, numbers FROM kl8_records ORDER BY period ASC", conn
                )
        if len(df) == 0:
            return None
        df['num_list'] = [
            [int(x) for x in str(s).split(',') if x.strip().isdigit()]
            for s in df['numbers']
        ]
        return df
    except Exception as e:
        log.error("加载快乐8数据失败: %s", e)
        return None


def ensure_kl8_data(num_periods=50, force_refresh=False):
    """
    确保快乐8真实数据可用（仅从500.com获取，不使用模拟数据）

    H4 修复: 除"条数够不够"外增加新鲜度校验——快乐8每日开1期，
    最新一期距今天超过 2 天即视为陈旧，强制重新拉取，避免用旧数据预测。
    """
    # 先检查数据库
    df = load_kl8_data(limit=num_periods)
    if df is not None and len(df) >= num_periods and not force_refresh:
        # 新鲜度校验：取最新一期日期（date 列或期号反推），超过 2 天视为陈旧
        latest = df.iloc[-1]
        latest_date = None
        ds = latest.get('date')
        if ds:
            try:
                latest_date = datetime.strptime(str(ds), '%Y-%m-%d').date()
            except ValueError:
                latest_date = None
        if latest_date is None:
            _d = period_to_date(latest.get('period', ''))
            if _d:
                try:
                    latest_date = datetime.strptime(_d, '%Y-%m-%d').date()
                except ValueError:
                    latest_date = None

        if latest_date is not None:
            days_old = (datetime.now().date() - latest_date).days
            if days_old <= 2:
                print(f"  从数据库加载 {len(df)} 期快乐8数据 (最新期 {latest['period']} · {latest_date} · {days_old}天前)")
                return df
            print(f"  ⚠️ 快乐8最新期 {latest['period']} 已是 {days_old} 天前，数据陈旧，重新拉取最新")
        else:
            # 极端兜底：无任何日期可校验时退化为仅按条数判断
            print(f"  从数据库加载 {len(df)} 期快乐8数据 (无日期信息，跳过新鲜度检查)")
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
