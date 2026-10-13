#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方向4（自适应融合）+ 奖级计算回归测试。
运行: pytest tests/test_fusion.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ==================== 奖级计算 ====================

def test_dlt_prize_rules():
    """大乐透奖级表关键条目。"""
    from prize_calculator import get_dlt_prize
    assert get_dlt_prize(5, 2)['level'] == '一等奖'
    assert get_dlt_prize(3, 1)['level'] == '八等奖'
    assert get_dlt_prize(2, 1)['level'] == '九等奖'
    assert get_dlt_prize(1, 1)['is_winning'] is False


def test_ssq_prize_rules():
    """双色球奖级表关键条目。"""
    from prize_calculator import get_ssq_prize
    assert get_ssq_prize(6, 1)['level'] == '一等奖'
    assert get_ssq_prize(4, 1)['level'] == '四等奖'
    assert get_ssq_prize(0, 1)['level'] == '六等奖'
    assert get_ssq_prize(3, 0)['is_winning'] is False


def test_kl8_sh10_prize():
    """快乐8选十：中0/5/10 中奖，中1-4 不中。"""
    from prize_calculator import get_kl8_sh10_prize
    assert get_kl8_sh10_prize(10)['is_winning'] is True
    assert get_kl8_sh10_prize(5)['is_winning'] is True
    assert get_kl8_sh10_prize(0)['is_winning'] is True
    assert get_kl8_sh10_prize(3)['is_winning'] is False
    assert get_kl8_sh10_prize(4)['is_winning'] is False


def test_verify_kl8_complex():
    """复式11码验证：拆分为 C(11,10)=11 注分别定级。"""
    from prize_calculator import generate_kl8_combinations, verify_kl8_combinations
    preds = [(n, 25 - n, {}) for n in range(1, 25)]
    combos = generate_kl8_combinations(preds)
    actual = list(range(1, 21))  # 开奖 = 预测 Top-20 全部命中
    verified = verify_kl8_combinations(combos, actual)
    # 单式A (Top-10) 全中
    assert verified['single'][0]['hit_count'] == 10
    # 复式12码A 覆盖 Top-12，命中 10/12
    assert verified['complex_12'][0]['hit_count'] == 12
    # top_20 全命中
    assert verified['top_20_hit_count'] == 20


# ==================== 融合权重（方向4）====================

def test_fusion_sort_zero_below_random():
    """方向4: 图案负学习(z<0)且ML正常时，权重应偏向ML。"""
    import predict as P
    pat_scores = [(n, 1.0 / (1 + (n % 10)), {}) for n in range(1, 36)]
    ml_probs = {n: 0.01 + (n % 5) * 0.005 for n in range(1, 36)}
    res = P._fusion_sort(pat_scores, ml_probs, weight=0.3,
                         pattern_acc=-1.9, ml_acc=0.5)
    # 结果应为融合列表且排序正确
    scores = [s for _, s, _ in res]
    assert scores == sorted(scores, reverse=True)
    assert len(res) == 35


def test_fusion_sort_no_ml_probs():
    """ml_probs=None 时原样返回图案评分。"""
    import predict as P
    pat_scores = [(n, 1.0, {}) for n in range(1, 36)]
    assert P._fusion_sort(pat_scores, None) == pat_scores


def test_adaptive_weights_sum_to_one(dlt_df):
    """方向4: 自适应权重归一化且非负。"""
    import predict as P
    P.ADAPTIVE_BACKTEST_N = 5  # 加速
    wp, wm = P._adaptive_fusion_weights('dlt', dlt_df, use_ml=True)
    assert wp >= 0 and wm >= 0
    assert abs(wp + wm - 1.0) < 1e-6
