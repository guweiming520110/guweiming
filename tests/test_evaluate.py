#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py 统计工具（超几何 z 检验）回归测试。
运行: pytest tests/test_evaluate.py -v
"""
import numpy as np
import pytest


def test_hypergeom_mean():
    """超几何期望: dlt 前区 5/35 抽 5，期望 = 5*5/35。"""
    from evaluate import hypergeom_mean
    assert abs(hypergeom_mean(5, 5, 35) - 5 * 5 / 35) < 1e-9
    assert abs(hypergeom_mean(20, 20, 80) - 20 * 20 / 80) < 1e-9


def test_hypergeom_var_positive():
    """超几何方差为正。"""
    from evaluate import hypergeom_var
    assert hypergeom_var(5, 5, 35) > 0


def test_zscore_random_level():
    """随机水平命中时 z 接近 0。"""
    from evaluate import zscore, hypergeom_mean
    obs = hypergeom_mean(5, 5, 35)  # 观测=理论 → z=0
    assert abs(zscore(obs, 5, 5, 35, 100)) < 1e-9


def test_zscore_positive_when_above():
    """高于随机时 z 为正。"""
    from evaluate import zscore
    # 观测命中率 1.0 vs 理论 0.714 → z 应显著为正
    z = zscore(1.0, 5, 5, 35, 60)
    assert z > 1.96


def test_zscore_negative_when_below():
    """低于随机时 z 为负。"""
    from evaluate import zscore
    z = zscore(0.5, 5, 5, 35, 60)
    assert z < -1.96
