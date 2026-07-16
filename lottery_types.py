#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数字彩票配置
定义各彩种的核心参数，走势图/预测系统自动适配
"""

LOTTERY_TYPES = {
    'dlt': {
        'name': '超级大乐透',
        'short_name': '大乐透',
        'front_name': '前区',
        'back_name': '后区',
        'front_min': 1, 'front_max': 35, 'front_count': 5,
        'back_min': 1, 'back_max': 12, 'back_count': 2,
        'has_back': True,
        'back_shift': 36,  # 前区35 + gap1 = 后区起始列
        'gap_cols': 1,
        'total_number_cols': 48,  # 35 + 1(gap) + 12
        'data_table': 'lottery_records',
        'data_type': 'dlt',
        'front_cols': ['front_1','front_2','front_3','front_4','front_5'],
        'back_cols': ['back_1','back_2'],
        'default_periods': 30,
        'default_train': 80,
        'color_ball': (0.1, 0.55, 0.1),  # 历史号码统一绿色
    },
    'kl8': {
        'name': '快乐8',
        'short_name': '快乐8',
        'front_name': '号码',
        'back_name': None,
        'front_min': 1, 'front_max': 80, 'front_count': 20,
        'back_min': 0, 'back_max': 0, 'back_count': 0,
        'has_back': False,
        'back_shift': 0,
        'gap_cols': 0,
        'total_number_cols': 80,
        'data_table': 'kl8_records',
        'data_type': 'kl8',
        'front_cols': None,  # kl8用num_list
        'back_cols': None,
        'default_periods': 20,
        'default_train': 50,
        'color_ball': (0.1, 0.55, 0.1),
    },
    'ssq': {
        'name': '双色球',
        'short_name': '双色球',
        'front_name': '红球',
        'back_name': '蓝球',
        'front_min': 1, 'front_max': 33, 'front_count': 6,
        'back_min': 1, 'back_max': 16, 'back_count': 1,
        'has_back': True,
        'back_shift': 34,  # 33 + gap1
        'gap_cols': 1,
        'total_number_cols': 50,  # 33 + 1(gap) + 16
        'data_table': 'lottery_records',
        'data_type': 'ssq',
        'front_cols': ['front_1','front_2','front_3','front_4','front_5','front_6'],
        'back_cols': ['back_1'],
        'default_periods': 30,
        'default_train': 80,
        'color_ball': (0.1, 0.55, 0.1),
    },
}

# 获取彩种配置
def get_config(lottery_type):
    """获取指定彩种的配置"""
    cfg = LOTTERY_TYPES.get(lottery_type)
    if not cfg:
        raise ValueError(f"不支持的彩种: {lottery_type}，可选: {list(LOTTERY_TYPES.keys())}")
    return cfg

# 列出所有支持的彩种
def list_types():
    return list(LOTTERY_TYPES.keys())

# 彩种中文名
def type_name(lottery_type):
    cfg = get_config(lottery_type)
    return cfg['name']


# 大乐透/双色球年度期数经验阈值：官方每年约 150-153 期，留余量取 153 作为跨年滚动阈值。
# ⚠️ 局限：这是基于历史规律的启发式推断，非精确日历。某年实际期数明显偏离 153 时，
# 年末那一期的推算可能偏差（例如该年只有 150 期却按 153 判断不滚动）。彻底根治需接官方
# 开奖日历；当前策略下，跨年前后应人工核对一次官方期号。真实期号始终以官方公布为准。
DLT_SSQ_YEARLY_MAX = 153


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def next_period(lottery_type, period):
    """
    根据当前期号推算下一期期号，正确处理跨年滚动。

    期号格式：
      - dlt/ssq: 5位 YYNNN（如 26072 = 2026年第72期）
      - kl8:     7位 YYYYNNN（如 2026170 = 2026年第170期），每天1期

    非法期号格式会抛 ValueError，避免静默产生错误期号后污染下游预测/验证。
    """
    period = str(period).strip()

    if lottery_type == 'kl8':
        # YYYYNNN：年份4位 + 序号（至少3位）
        if len(period) < 5 or not period.isdigit():
            raise ValueError(f"快乐8期号格式非法（应为 YYYYNNN，如 2026170）: {period!r}")
        year = int(period[:4])
        seq = int(period[4:])
        # 快乐8每天开一期，平年365期/闰年366期，超过即滚动到次年
        days_in_year = 366 if _is_leap_year(year) else 365
        if seq >= days_in_year:
            return f"{year + 1}001"
        return f"{year}{seq + 1:03d}"

    # dlt / ssq：YYNNN
    if len(period) != 5 or not period.isdigit():
        raise ValueError(f"{lottery_type} 期号格式非法（应为 5 位 YYNNN，如 26072）: {period!r}")
    year = int(period[:2])
    seq = int(period[2:])
    if seq >= DLT_SSQ_YEARLY_MAX:
        return f"{year + 1:02d}001"
    return f"{year:02d}{seq + 1:03d}"


if __name__ == '__main__':
    for k, v in LOTTERY_TYPES.items():
        print(f"{k}: {v['name']} ({v['front_name']}{v['front_min']}-{v['front_max']}选{v['front_count']})")
