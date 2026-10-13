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


def test_composite_chart_output(tmp_path, dlt_df):
    """合成走势大图：输出到临时目录并生成有效 PNG（计划任务实际路径）。"""
    from charts.composite import generate_composite_chart
    out = generate_composite_chart(
        dlt_df, lottery_type='dlt',
        network_periods=min(10, len(dlt_df)),
        trend_periods=min(15, len(dlt_df)),
        freq_periods=min(20, len(dlt_df)),
        output_dir=tmp_path,
        predicted_front=[1, 5, 10, 20, 30],
        predicted_back=[3, 8],
    )
    assert out is not None
    assert out.exists()
    # 验证是合法 PNG（文件头签名）
    with open(out, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'


def test_composite_chart_failure_returns_none(dlt_df):
    """数据异常时合成图返回 None 而非抛异常/残留 buffer。"""
    from charts.composite import generate_composite_chart
    import pandas as pd
    empty = pd.DataFrame(columns=dlt_df.columns)
    out = generate_composite_chart(empty, lottery_type='dlt', output_dir=None)
    assert out is None
