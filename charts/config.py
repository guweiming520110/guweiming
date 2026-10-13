#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
走势图布局配置 — 从 lottery_types 配置计算绘图参数
"""

from lottery_types import get_config

# M9 修复: 中文字体统一下沉到 charts/config.py（模块加载即生效），
# 避免只在 generate_charts.py 设置——run_all 等入口直接调用 charts.* 模块时
# 不经过 generate_charts，中文会缺字/方框。
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def chart_config(lottery_type):
    """
    从彩种配置计算走势图布局参数

    返回 dict 包含:
      - 原始配置 (name, front_name, back_name, etc.)
      - 前区列范围 FRONT_MIN~FRONT_MAX
      - gap列范围 GAP_START~GAP_END
      - 后区列范围 BACK_START~BACK_END (仅 has_back=True)
      - 总列数 TOTAL_COLS
      - 后列偏移量 BACK_SHIFT
      - 用于走势图颜色分区的区域划分
    """
    cfg = get_config(lottery_type).copy()
    fm = cfg['front_min']
    fx = cfg['front_max']
    gap = cfg['gap_cols']
    bs = cfg['back_shift']          # 后区起始列 = bs + back_min (已含gap)
    bm = cfg['back_min']
    bx = cfg['back_max']

    total_cols = cfg['total_number_cols']

    # 各区域在绘图矩阵中的列范围
    zones = []
    if fm <= fx:
        zones.append(('front', fm, fx))

    gap_start = fx + 1
    gap_end = fx + gap
    if gap > 0:
        zones.append(('gap', gap_start, gap_end))

    if cfg.get('has_back', False):
        back_start = bs + bm  # 实际绘图列
        back_end = bs + bx
        zones.append(('back', back_start, back_end))
    else:
        back_start = None
        back_end = None

    result = {
        'lottery_type': lottery_type,
        'name': cfg['name'],
        'short_name': cfg.get('short_name', cfg['name']),
        'front_name': cfg['front_name'],
        'back_name': cfg.get('back_name'),
        'front_min': fm,
        'front_max': fx,
        'front_count': cfg['front_count'],
        'back_min': bm,
        'back_max': bx,
        'back_count': cfg['back_count'],
        'has_back': cfg.get('has_back', False),
        'gap_cols': gap,
        'back_shift': bs,
        'total_cols': total_cols,
        'gap_start': gap_start,
        'gap_end': gap_end,
        'back_start': back_start,
        'back_end': back_end,
        'zones': zones,            # [('front',1,35), ('gap',36,36), ('back',37,48)]
        'front_cols': cfg.get('front_cols'),
        'back_cols': cfg.get('back_cols'),
        'data_table': cfg['data_table'],
        'data_type': cfg['data_type'],
        'default_periods': cfg.get('default_periods', 30),
        'default_train': cfg.get('default_train', 80),
        'color_ball': cfg.get('color_ball', (0.1, 0.55, 0.1)),
    }
    return result


def zone_color_scheme(lottery_type):
    """返回前区底色分区配置 (仅用于 dlt/ssq 等有前区分段意义的彩种)"""
    schemes = {
        'dlt': [
            (1, 12, '#4CAF50', 0.025),
            (13, 24, '#FFC107', 0.025),
            (25, 35, '#F44336', 0.025),
        ],
        'ssq': [
            (1, 11, '#4CAF50', 0.025),
            (12, 22, '#FFC107', 0.025),
            (23, 33, '#F44336', 0.025),
        ],
    }
    return schemes.get(lottery_type, [])


def future_periods_count():
    """未来预测期数（走势图底部留空行）"""
    return 3


# 历史号码球颜色常量
# L7 修复: 与 lottery_types.LOTTERY_TYPES[*]['color_ball'] 重复定义，改为从单一真源读取，
# 避免两处维护不同步。
from lottery_types import LOTTERY_TYPES
HISTORY_BALL_COLOR = LOTTERY_TYPES['dlt'].get('color_ball', (0.1, 0.55, 0.1))
