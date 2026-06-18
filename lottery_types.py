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

if __name__ == '__main__':
    for k, v in LOTTERY_TYPES.items():
        print(f"{k}: {v['name']} ({v['front_name']}{v['front_min']}-{v['front_max']}选{v['front_count']})")
