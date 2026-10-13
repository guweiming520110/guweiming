#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图表模块回归测试。
运行: pytest tests/test_charts.py -v
"""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # 无头模式，避免弹窗/阻塞


def test_chart_config_dlt():
    """charts/config: dlt 布局配置正确（前区1-35/gap/后区）。"""
    from charts.config import chart_config
    cfg = chart_config('dlt')
    assert cfg['front_min'] == 1 and cfg['front_max'] == 35
    assert cfg['back_start'] == 37  # 35 + 1(gap) + 1
    assert cfg['total_cols'] == 48
    assert cfg['has_back'] is True


def test_chart_config_kl8():
    """charts/config: kl8 无后区、80 列。"""
    from charts.config import chart_config
    cfg = chart_config('kl8')
    assert cfg['front_max'] == 80
    assert cfg['has_back'] is False
    assert cfg['total_cols'] == 80


def test_get_ball_color(dlt_df):
    """charts/utils: get_ball_color 返回合法 RGB。"""
    from charts.utils import get_ball_color
    from charts.config import chart_config
    cfg = chart_config('dlt')
    c = get_ball_color(10, cfg)
    assert len(c) == 3 and all(0 <= x <= 1 for x in c)


def test_draw_number_ball(dlt_df):
    """charts/utils: 画号码球不抛异常。"""
    import matplotlib.pyplot as plt
    from charts.utils import draw_number_ball
    fig, ax = plt.subplots()
    draw_number_ball(ax, 5, 1, 5, (0.1, 0.5, 0.1))
    plt.close(fig)


def test_trend_chart_generation(dlt_df):
    """charts/trend: 生成趋势图不抛异常。"""
    from charts.trend import generate_trend_chart
    out = generate_trend_chart(dlt_df, lottery_type='dlt')
    assert out is not None


def test_frequency_chart_generation(dlt_df):
    """charts/frequency: 频次图不抛异常。"""
    from charts.frequency import generate_frequency_chart
    out = generate_frequency_chart(dlt_df, lottery_type='dlt')
    assert out is not None
