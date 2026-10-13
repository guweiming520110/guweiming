#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心回归测试：覆盖第一轮修复的关键缺陷。
运行: pytest tests/test_core_fixes.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ==================== S1: SSQ 深度学习号码范围 ====================

def test_dl_ssq_range():
    """S1: ssq 构造后号码范围应为 33/16（此前沿用 dlt 默认 35/12）。"""
    from deep_learning_models import DeepLearningPredictor
    p = DeepLearningPredictor(lottery_type='ssq')
    assert p.front_max == 33
    assert p.back_max == 16


def test_dl_load_models_rebuilds_range(tmp_path):
    """S1: load_models 恢复彩种后必须重建号码范围。"""
    import pickle
    from deep_learning_models import DeepLearningPredictor
    cfg = tmp_path / 'dl_config.pkl'
    with open(cfg, 'wb') as f:
        pickle.dump({'sequence_length': 10, 'lottery_type': 'ssq'}, f)
    p = DeepLearningPredictor()  # 默认 dlt
    assert p.front_max == 35
    p.load_models(str(tmp_path))  # 无模型文件返回 False，但已重建范围
    assert p.lottery_type == 'ssq'
    assert p.front_max == 33
    assert p.back_max == 16


# ==================== C1: 空库 pattern_learner 崩溃 ====================

def test_pattern_learner_empty_db_no_crash(empty_dlt_df):
    """C1: 空库时 predict 不应抛 IndexError，返回空列表。"""
    from pattern_learner.core import PatternLearner
    pl = PatternLearner('dlt')
    pl.train(empty_dlt_df, num_periods=30)
    preds, report = pl.predict_with_report(top_k=5)
    assert preds == []
    assert isinstance(report, str)


# ==================== C2: should_connect step=2 隔空连线 ====================

def _matrix_with(points, rows=30, cols=50):
    m = [[False] * cols for _ in range(rows)]
    for r, c in points:
        m[r][c] = True
    return m


def test_should_connect_step2_arithmetic():
    """C2: 隔1空(等差)三连应连线，非等差不连。"""
    from charts.utils import should_connect
    # 等差: (0,5),(2,5),(4,5) → 应 True
    m = _matrix_with([(0, 5), (2, 5), (4, 5)])
    assert should_connect(m, 30, 35, 0, 5, 2, 5, step=2) is True
    assert should_connect(m, 30, 35, 2, 5, 4, 5, step=2) is True
    # 非等差: (0,5),(2,5),(5,5) → 应 False
    m2 = _matrix_with([(0, 5), (2, 5), (5, 5)])
    assert should_connect(m2, 30, 35, 0, 5, 2, 5, step=2) is False


def test_should_connect_step1_regression():
    """C2: step=1 常规三连不回归。"""
    from charts.utils import should_connect
    m = _matrix_with([(0, 5), (0, 6), (0, 7)])
    assert should_connect(m, 30, 35, 0, 5, 0, 6, step=1) is True


# ==================== M10: 嵌套图案去重 ====================

def test_nested_pattern_no_double_count():
    """M10: 垂直4连不应同时计垂直3连。"""
    from pattern_learner.patterns import count_patterns_around
    mat = np.zeros((6, 12), dtype=bool)
    for i in range(4):
        mat[i][5] = True
    counts = count_patterns_around(mat, 0, 5)
    assert counts.get('vert_line_4', 0) == 1
    assert counts.get('vert_line_3', 0) == 0


def test_nested_pattern_3line_still_counts():
    """M10: 仅垂直3连时仍计 vert_line_3。"""
    from pattern_learner.patterns import count_patterns_around
    mat = np.zeros((6, 12), dtype=bool)
    for i in range(3):
        mat[i][5] = True
    counts = count_patterns_around(mat, 0, 5)
    assert counts.get('vert_line_3', 0) == 1
    assert counts.get('vert_line_4', 0) == 0


# ==================== 改进1: 跨年期号推算 ====================

def test_next_period_dlt_2025_actual():
    """改进1: dlt 2025 年实际 150 期，25150 之后应为 26001。"""
    from lottery_types import next_period
    assert next_period('dlt', '25150') == '26001'
    assert next_period('dlt', '25149') == '25150'


def test_next_period_ssq_2025_actual():
    """改进1: ssq 2025 年实际 151 期，25151 之后应为 26001。"""
    from lottery_types import next_period
    assert next_period('ssq', '25151') == '26001'
    assert next_period('ssq', '25150') == '25151'


def test_next_period_kl8_leap():
    """kl8: 平年 365 期/闰年 366 期跨年。"""
    from lottery_types import next_period
    assert next_period('kl8', '2026365') == '2027001'
    assert next_period('kl8', '2025001') == '2025002'


# ==================== 改进2: kl8 组合覆盖 ====================

def test_kl8_combinations_coverage():
    """改进2: 6 组注应覆盖 Top-20 全部号码（旧实现仅 14）。"""
    from prize_calculator import generate_kl8_combinations
    preds = [(n, 25 - n, {}) for n in range(1, 25)]
    combos = generate_kl8_combinations(preds)
    covered = set()
    for key in ('single', 'complex_11', 'complex_12'):
        for c in combos[key]:
            covered |= set(c)
    assert len(covered) == 20
    # 单式 A = Top-10, 单式 B = Top-11~20
    assert combos['single'][0] == sorted(range(1, 11))
    assert combos['single'][1] == sorted(range(11, 21))
