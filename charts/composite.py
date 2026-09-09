#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合成大图 — 将网络图+趋势图+频率图组合为一张大图
支持 dlt/kl8/ssq 等彩种
"""

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from charts.network import generate_network_chart
from charts.trend import generate_trend_chart
from charts.frequency import generate_frequency_chart
from charts.config import chart_config


def generate_composite_chart(df, lottery_type='dlt',
                             network_periods=30, trend_periods=50,
                             freq_periods=100, output_dir=None,
                             predicted_front=None, predicted_back=None,
                             verified_front=None, verified_back=None,
                             predicted_reasons=None):
    """
    统一合成走势大图

    参数:
        df: 数据 DataFrame
        lottery_type: 彩种类型 (dlt/kl8/ssq)
        network_periods: 网络图期数
        trend_periods: 趋势图期数
        freq_periods: 频率图期数
        output_dir: 输出目录
        其他: 预测/验证相关数据
    """
    cfg = chart_config(lottery_type)
    name = cfg['name']

    if output_dir is None:
        BASE_DIR = Path(__file__).resolve().parent.parent
        output_dir = BASE_DIR / 'charts'
    output_dir = Path(output_dir)
    os.makedirs(str(output_dir), exist_ok=True)

    total = len(df)
    show_net = min(network_periods, total)
    show_trend = min(trend_periods, total)
    show_freq = min(freq_periods, total)

    print(f"彩种: {name}")
    print(f"总计 {total} 期")
    print(f"  连通网络图: {show_net}期")
    print(f"  趋势图: {show_trend}期")
    print(f"  频率图: {show_freq}期")

    # 预测数据
    pf = predicted_front if predicted_front else []
    pb = predicted_back if predicted_back else []
    # 保留 None 语义：None=未验证, []或非空列表=已验证
    vf = verified_front  # 保持原样传递 (None/set/list)
    vb = verified_back   # 保持原样传递 (None/set/list)

    # 判断是否 kl8（独有布局：3张子图高度比不同）
    is_kl8 = (lottery_type == 'kl8')

    generators = [
        ("network", f"{name}连通网络图",
         lambda: generate_network_chart(
             df, lottery_type=lottery_type, num_periods=show_net,
             predicted_front=pf, predicted_back=pb,
             verified_front=vf, verified_back=vb,
             predicted_reasons=predicted_reasons)),
        ("trend", f"{name}趋势图",
         lambda: generate_trend_chart(
             df, lottery_type=lottery_type, num_periods=show_trend,
             predicted_front=pf, predicted_back=pb,
             verified_front=vf, verified_back=vb)),
        ("freq", f"{name}频率分布图",
         lambda: generate_frequency_chart(df, lottery_type=lottery_type, num_periods=show_freq)),
    ]

    buffers = {}
    failed = []
    for name_key, title, gen in generators:
        print(f"  {title} ...", end=' ')
        fig = None
        try:
            fig = gen()
            if fig:
                buf = io.BytesIO()
                dpi = 150
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                buf.seek(0)
                buffers[name_key] = buf
                print("OK")
            else:
                print("跳过")
        except Exception as e:
            failed.append(f"{title}: {e}")
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 无论成功失败都必须关闭，否则异常路径下 pyplot 会持续累积 figure
            if fig is not None:
                plt.close(fig)

    if not buffers:
        print("没有生成任何图表！")
        return None

    if failed:
        # 残缺的合成图会让人误以为分析完整，宁可不输出
        print(f"  ⚠️ {len(failed)} 张子图生成失败，放弃输出残缺合成图")
        for f in failed:
            print(f"     - {f}")
        return None

    images = {k: plt.imread(buf) for k, buf in buffers.items()}

    # 合成大图
    if is_kl8:
        fig = plt.figure(figsize=(30, 22), facecolor='#F5F5F5')
        gs = GridSpec(3, 1, height_ratios=[1.8, 1.2, 0.6], hspace=0.08)
        output_filename = 'kl8_analysis.png'
    else:
        fig = plt.figure(figsize=(28, 22), facecolor='#F5F5F5')
        gs = GridSpec(3, 1, height_ratios=[2.0, 1.0, 0.6], hspace=0.08)
        output_filename = f'{lottery_type}_analysis.png'

    for idx, (name_key, row) in enumerate([("network", 0), ("trend", 1), ("freq", 2)]):
        if name_key not in images:
            continue
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(images[name_key])
        ax.axis('off')

    fig.suptitle(f'{name} · 综合走势分析图',
                 fontsize=16, fontweight='bold', y=0.98, color='#333333')
    fig.text(0.5, 0.005,
             f'数据来源: lottery.db | 共 {total} 期历史数据 | '
             f'生成: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
             ha='center', va='bottom', fontsize=8, color='#999999')

    output_path = output_dir / output_filename
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    size_kb = output_path.stat().st_size // 1024
    print(f"\n{'=' * 50}")
    print(f"  合成大图: {output_path.resolve()}")
    print(f"  大小: {size_kb} KB")
    print(f"{'=' * 50}")

    for buf in buffers.values():
        buf.close()

    return output_path
