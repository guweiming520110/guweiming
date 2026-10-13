#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共享 pytest fixtures：在内存中构造各彩种测试数据，不依赖真实数据库。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保可导入项目模块
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _make_dlt_df(n=100, seed=42):
    """构造确定性 dlt 数据：period 升序、前区 1-35 升序不重复、后区 1-12。"""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        period = f"25{i + 1:03d}"
        front = sorted(rng.choice(range(1, 36), 5, replace=False))
        back = sorted(rng.choice(range(1, 13), 2, replace=False))
        rows.append([period, f"2025-01-{(i % 28) + 1:02d}"] + front + back)
    return pd.DataFrame(rows, columns=[
        'period', 'date', 'front_1', 'front_2', 'front_3', 'front_4', 'front_5', 'back_1', 'back_2'])


def _make_ssq_df(n=100, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        period = f"25{i + 1:03d}"
        front = sorted(rng.choice(range(1, 34), 6, replace=False))
        back = int(rng.choice(range(1, 17)))
        rows.append([period, f"2025-01-{(i % 28) + 1:02d}"] + front + [back])
    return pd.DataFrame(rows, columns=[
        'period', 'date', 'front_1', 'front_2', 'front_3', 'front_4', 'front_5', 'front_6', 'back_1'])


def _make_kl8_df(n=100, seed=9):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        period = f"2026{i + 1:03d}"
        nums = sorted(rng.choice(range(1, 81), 20, replace=False))
        rows.append([period, f"2026-01-{(i % 28) + 1:02d}", nums])
    return pd.DataFrame(rows, columns=['period', 'date', 'num_list'])


@pytest.fixture
def dlt_df():
    return _make_dlt_df()


@pytest.fixture
def ssq_df():
    return _make_ssq_df()


@pytest.fixture
def kl8_df():
    return _make_kl8_df()


@pytest.fixture
def empty_dlt_df():
    """空库场景（首启无数据）"""
    return pd.DataFrame(columns=['period', 'date', 'front_1', 'front_2', 'front_3',
                                 'front_4', 'front_5', 'back_1', 'back_2'])
