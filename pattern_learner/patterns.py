#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
几何图案定义与匹配模块

每种图案定义为相对于锚点(0,0)的坐标偏移列表。
坐标：(行偏移, 列偏移) 行=期数方向(下为正), 列=号码方向(右为正)
"""

import numpy as np
from collections import defaultdict

# ==================== 图案模板定义 ====================
# 每个图案: (name, [(dr, dc), ...], min_periods, max_gap)
# dr=行偏移(期), dc=列偏移(号)
# min_periods=至少跨越几期(用于筛选)
# max_gap=最大步长(1=相邻, 2=空1格)

PATTERN_TEMPLATES = [
    # ---- 1. 水平直线（C级：最普通，几乎无预测价值） ----
    {
        'name': 'horiz_line_3',
        'label': '水平3连',
        'offsets': [(0, 0), (0, 1), (0, 2)],
        'step': 1,
        'min_rows': 1,
        'complexity': 'C',
        'base_score': 0.3,
    },
    {
        'name': 'horiz_line_4',
        'label': '水平4连',
        'offsets': [(0, 0), (0, 1), (0, 2), (0, 3)],
        'step': 1,
        'min_rows': 1,
        'complexity': 'C',
        'base_score': 0.4,
    },
    {
        'name': 'horiz_gap2',
        'label': '水平隔1空3连',
        'offsets': [(0, 0), (0, 2), (0, 4)],
        'step': 1,
        'min_rows': 1,
        'complexity': 'C',
        'base_score': 0.4,
    },

    # ---- 2. 垂直直线（B级：跨期形成，有一定价值） ----
    {
        'name': 'vert_line_3',
        'label': '垂直3连',
        'offsets': [(0, 0), (1, 0), (2, 0)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'B',
        'base_score': 0.8,
    },
    # 注：原 vert_symmetry(垂直对称) 与 vert_line_3 均为竖直3连，去平移后footprint完全相同，已删除以消除双重计分
    {
        'name': 'vert_line_4',
        'label': '垂直4连',
        'offsets': [(0, 0), (1, 0), (2, 0), (3, 0)],
        'step': 1,
        'min_rows': 4,
        'complexity': 'B',
        'base_score': 1.2,
    },
    # 注：原 vert_gap2(垂直隔1空3连) 与 skip1_3vert 去平移后 footprint 完全相同
    # （step 修复后均为偏移 0,2,4），已删除以免双重计分；语义由 skip1_3vert 承担

    # ---- 3. 对角线 ↘（B级） ----
    {
        'name': 'diag_se_3',
        'label': '对角↘3连',
        'offsets': [(0, 0), (1, 1), (2, 2)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'B',
        'base_score': 0.8,
    },
    # 注：原 diag_se_gap2 与 skip1_diag 去平移后 footprint 完全相同（step 修复后均为偏移 0,2,4），
    # 已删除以免双重计分；隔1空对角语义由 skip1_diag 承担

    # ---- 4. 对角线 ↙（B级） ----
    {
        'name': 'diag_sw_3',
        'label': '对角↙3连',
        'offsets': [(0, 0), (1, -1), (2, -2)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'B',
        'base_score': 0.8,
    },
    {
        'name': 'diag_sw_gap2',
        'label': '对角↙隔1空',
        'offsets': [(0, 0), (2, -2), (4, -4)],
        'step': 1,
        'min_rows': 5,
        'complexity': 'B',
        'base_score': 0.9,
    },

    # ---- 5. 方块（S级：多点配合，稀有） ----
    {
        'name': 'square_2x2',
        'label': '2×2方块',
        'offsets': [(0, 0), (0, 1), (1, 0), (1, 1)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'S',
        'base_score': 2.0,
    },
    {
        'name': 'square_2x2_gap',
        'label': '2×2隔空方块',
        'offsets': [(0, 0), (0, 2), (2, 0), (2, 2)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'S',
        'base_score': 2.5,
    },

    # ---- 6. 三角形L型（A级：需要L型排列） ----
    {
        'name': 'triangle_bl',
        'label': '三角┘型',
        'offsets': [(0, 0), (0, 1), (1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.2,
    },
    {
        'name': 'triangle_br',
        'label': '三角└型',
        'offsets': [(0, 0), (0, -1), (1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.2,
    },
    {
        'name': 'triangle_tl',
        'label': '三角┐型',
        'offsets': [(0, 0), (0, 1), (-1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.2,
    },
    {
        'name': 'triangle_tr',
        'label': '三角┌型',
        'offsets': [(0, 0), (0, -1), (-1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.2,
    },

    # ---- 7. 十字形（S级：需中心+4臂，极稀有） ----
    {
        'name': 'cross_4',
        'label': '十字4臂',
        'offsets': [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'S',
        'base_score': 3.0,
    },

    # ---- 8. T型（A级） ----
    # 注：原 cross_3(十字3臂) 与 t_shape_down 形状完全相同，已删除以消除双重计分
    {
        'name': 't_shape_down',
        'label': 'T型向下',
        'offsets': [(0, 0), (0, 1), (0, -1), (1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.5,
    },
    {
        'name': 't_shape_up',
        'label': 'T型向上',
        'offsets': [(0, 0), (0, 1), (0, -1), (-1, 0)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.5,
    },

    # ---- 9. 对角线扩展（B级） ----
    {
        'name': 'diag_se_4',
        'label': '对角↘4连',
        'offsets': [(0, 0), (1, 1), (2, 2), (3, 3)],
        'step': 1,
        'min_rows': 4,
        'complexity': 'B',
        'base_score': 1.5,
    },
    {
        'name': 'diag_sw_4',
        'label': '对角↙4连',
        'offsets': [(0, 0), (1, -1), (2, -2), (3, -3)],
        'step': 1,
        'min_rows': 4,
        'complexity': 'B',
        'base_score': 1.5,
    },
    {
        'name': 'diag_se_gap3',
        'label': '对角↘隔2空',
        'offsets': [(0, 0), (3, 3), (6, 6)],
        'step': 1,
        'min_rows': 7,
        'complexity': 'B',
        'base_score': 1.0,
    },
    {
        'name': 'diag_sw_gap3',
        'label': '对角↙隔2空',
        'offsets': [(0, 0), (3, -3), (6, -6)],
        'step': 1,
        'min_rows': 7,
        'complexity': 'B',
        'base_score': 1.0,
    },

    # ---- 10. 折线图案（A级） ----
    {
        'name': 'z_shape',
        'label': 'Z字形',
        'offsets': [(0, 0), (0, 1), (1, 1), (1, 2)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'A',
        'base_score': 1.4,
    },
    {
        'name': 'n_shape',
        'label': 'N字形',
        'offsets': [(0, 0), (1, 0), (1, 1), (2, 1)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'A',
        'base_score': 1.4,
    },
    # 注：原 step_up 与 diag_se_3 形状完全相同，已删除以消除双重计分
    # 注：原 step_down 与 diag_sw_3 形状完全相同（去平移后footprint一致），已删除以消除双重计分

    # ---- 11. 曲线弧形（S级） ----
    {
        'name': 'arc_up_3',
        'label': '上凸弧3点',
        'offsets': [(0, 1), (1, 0), (1, 2)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'S',
        'base_score': 2.2,
    },
    {
        'name': 'arc_down_3',
        'label': '下凹弧3点',
        'offsets': [(0, 0), (0, 2), (1, 1)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'S',
        'base_score': 2.2,
    },
    {
        'name': 'arc_up_4',
        'label': '上凸弧4点',
        'offsets': [(0, 2), (1, 1), (1, 3), (2, 2)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'S',
        'base_score': 2.8,
    },
    # 注：原 arc_down_4(下凹弧4点) 与 arc_up_4 均为同一菱形，去平移后footprint完全相同，已删除以消除双重计分

    # ---- 12. 对称图案（S级） ----
    {
        'name': 'horiz_symmetry',
        'label': '水平对称',
        'offsets': [(0, -2), (0, 0), (0, 2)],
        'step': 1,
        'min_rows': 1,
        'complexity': 'S',
        'base_score': 1.8,
    },
    # 注：原 vert_symmetry(垂直对称) 已上移合并至 vert_line_3 处删除，见上文说明
    {
        'name': 'x_symmetry',
        'label': '点对称X形',
        'offsets': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'S',
        'base_score': 3.0,
    },

    # ---- 13. 间隔模式（A级） ----
    {
        'name': 'skip1_3vert',
        'label': '跳1竖连3',
        'offsets': [(0, 0), (2, 0), (4, 0)],
        'step': 1,
        'min_rows': 5,
        'complexity': 'A',
        'base_score': 1.1,
    },
    {
        'name': 'skip2_3vert',
        'label': '跳2竖连3',
        'offsets': [(0, 0), (3, 0), (6, 0)],
        'step': 1,
        'min_rows': 7,
        'complexity': 'A',
        'base_score': 1.0,
    },
    {
        'name': 'skip1_diag',
        'label': '跳1斜连3',
        'offsets': [(0, 0), (2, 2), (4, 4)],
        'step': 1,
        'min_rows': 5,
        'complexity': 'A',
        'base_score': 1.1,
    },

    # ---- 14. 复合矩形/角落（A级/S级） ----
    {
        'name': 'rect_2x3',
        'label': '2×3矩形',
        'offsets': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        'step': 1,
        'min_rows': 2,
        'complexity': 'S',
        'base_score': 2.5,
    },
    {
        'name': 'rect_3x2',
        'label': '3×2矩形',
        'offsets': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        'step': 1,
        'min_rows': 3,
        'complexity': 'S',
        'base_score': 2.5,
    },
    # 注：原 corner_missing_tl 与 triangle_tr 形状完全相同（去平移后footprint一致），已删除以消除双重计分
    # 注：原 corner_missing_tr 与 triangle_tl 形状完全相同（去平移后footprint一致），已删除以消除双重计分
    # 注：原 corner_missing_bl 与 triangle_br 形状完全相同（去平移后footprint一致），已删除以消除双重计分
    # 注：原 corner_missing_br 与 triangle_bl 形状完全相同，已删除以消除双重计分
]

# ==================== 图案匹配引擎 ====================


def match_pattern_at(matrix, pattern, row, col):
    """
    检查在矩阵的(row, col)位置是否存在指定图案

    参数:
        matrix - (n_periods, max_num+1) 布尔矩阵
        pattern - 图案模板字典
        row, col - 检查位置

    返回:
        bool - 是否匹配
    """
    n_periods, n_cols = matrix.shape
    max_num = n_cols - 1
    step = pattern.get('step', 1)

    for dr, dc in pattern['offsets']:
        r = row + dr * step
        c = col + dc * step
        if r < 0 or r >= n_periods or c < 1 or c > max_num:
            return False
        if not matrix[r][c]:
            return False
    return True


def find_all_patterns(matrix, patterns=None):
    """
    在矩阵中找出所有出现的图案

    参数:
        matrix - 布尔矩阵
        patterns - 要检查的图案列表(默认全部)

    返回:
        dict: {pattern_name: [(row, col), ...]}
    """
    if patterns is None:
        patterns = PATTERN_TEMPLATES

    n_periods = matrix.shape[0]
    max_num = matrix.shape[1] - 1

    results = defaultdict(list)

    for pattern in patterns:
        name = pattern['name']
        max_row = n_periods - 1
        max_col = max_num

        for row in range(max_row + 1):
            for col in range(1, max_col + 1):
                if match_pattern_at(matrix, pattern, row, col):
                    results[name].append((row, col))

    return dict(results)


def count_patterns_around(matrix, row, col, patterns=None):
    """
    统计某个位置参与了多少个图案

    参数:
        matrix - 布尔矩阵
        row, col - 目标位置(通常是候选号码位置)
        patterns - 图案列表

    返回:
        dict: {pattern_name: count}
    """
    if patterns is None:
        patterns = PATTERN_TEMPLATES

    counts = defaultdict(int)
    n_periods, n_cols = matrix.shape
    max_num = n_cols - 1

    for pattern in patterns:
        name = pattern['name']
        step = pattern.get('step', 1)
        offsets = pattern['offsets']

        for anchor_idx, (dr0, dc0) in enumerate(offsets):
            anchor_row = row - dr0 * step
            anchor_col = col - dc0 * step

            if anchor_row < 0 or anchor_col < 1:
                continue
            if anchor_row >= n_periods or anchor_col > max_num:
                continue

            valid = True
            for dr, dc in offsets:
                r = anchor_row + dr * step
                c = anchor_col + dc * step
                if r < 0 or r >= n_periods or c < 1 or c > max_num:
                    valid = False
                    break
                if not matrix[r][c]:
                    valid = False
                    break

            if valid:
                counts[name] += 1

    return dict(counts)


def count_patterns_around_weighted(matrix, row, col, patterns=None, decay=0.95):
    """
    统计图案并应用时间衰减（越远的图案权重越低）

    参数:
        matrix - 布尔矩阵
        row, col - 目标位置
        patterns - 图案列表
        decay - 衰减系数 (每期衰减, 0.95≈50期后≈8%)

    返回:
        dict: {pattern_name: (count, weighted_sum)}
    """
    if patterns is None:
        patterns = PATTERN_TEMPLATES

    result = {}
    n_periods, n_cols = matrix.shape
    max_num = n_cols - 1

    for pattern in patterns:
        name = pattern['name']
        step = pattern.get('step', 1)
        offsets = pattern['offsets']
        total_count = 0
        total_weighted = 0.0

        for anchor_idx, (dr0, dc0) in enumerate(offsets):
            anchor_row = row - dr0 * step
            anchor_col = col - dc0 * step

            if anchor_row < 0 or anchor_col < 1:
                continue
            if anchor_row >= n_periods or anchor_col > max_num:
                continue

            valid = True
            # 行号越大表示期数越久远(row=0为最新一期)，用图案触及的最旧行(max_row)作为衰减基准，
            # 使跨越更久远历史的图案权重更低，与"近期图案权重更高"的语义一致
            max_row = 0
            for dr, dc in offsets:
                r = anchor_row + dr * step
                c = anchor_col + dc * step
                if r < 0 or r >= n_periods or c < 1 or c > max_num:
                    valid = False
                    break
                if not matrix[r][c]:
                    valid = False
                    break
                if r > max_row:
                    max_row = r

            if valid:
                total_count += 1
                total_weighted += decay ** max_row

        # 仅在实际匹配到图案时写入，保持与 count_patterns_around 的语义一致
        if total_count > 0:
            result[name] = (total_count, round(total_weighted, 4))

    return result


def get_pattern_label(name):
    """根据图案名称获取中文标签"""
    for p in PATTERN_TEMPLATES:
        if p['name'] == name:
            return p['label']
    return name


def list_pattern_types():
    """列出所有图案类型"""
    return [(p['name'], p['label']) for p in PATTERN_TEMPLATES]
