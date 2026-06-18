#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
号码频率分布图 — 配置驱动，支持有后区(dlt/ssq)和无后区(kl8)彩种
"""

import numpy as np
import matplotlib.pyplot as plt
from charts.utils import get_ball_color
from charts.config import chart_config


def generate_frequency_chart(df, lottery_type='dlt', num_periods=100, figsize=(24, 8)):
    """统一号码频率分布图"""
    cfg = chart_config(lottery_type)
    data = df.tail(num_periods).copy()
    n = len(data)
    if n == 0:
        return None

    has_back = cfg['has_back']
    front_min, front_max = cfg['front_min'], cfg['front_max']
    back_min, back_max = cfg['back_min'], cfg['back_max']

    if has_back:
        # 有后区：左右并排 前区频率 + 后区频率
        front_freq, back_freq = {}, {}
        front_cols = cfg['front_cols']
        back_cols = cfg['back_cols']
        for col in front_cols:
            for _, row in data.iterrows():
                num = int(row[col])
                front_freq[num] = front_freq.get(num, 0) + 1
        for col in back_cols:
            for _, row in data.iterrows():
                num = int(row[col])
                back_freq[num] = back_freq.get(num, 0) + 1

        fig, (ax_front, ax_back) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [3, 1]},
            facecolor='#FAFAFA'
        )

        # 前区柱状图
        front_nums = list(range(front_min, front_max + 1))
        front_counts = [front_freq.get(n, 0) for n in front_nums]
        ax_front.bar(front_nums, front_counts,
                     color=[get_ball_color(n, cfg) for n in front_nums],
                     edgecolor='white', linewidth=0.8, alpha=0.85, zorder=3)
        for num, count in zip(front_nums, front_counts):
            if count > 0:
                ax_front.text(num, count + 0.1, str(count), ha='center', va='bottom',
                              fontsize=7, fontweight='bold', zorder=4)
        ax_front.set_xlim(front_min - 1, front_max + 1)
        ax_front.set_xticks(range(front_min, front_max + 1, 2))
        ax_front.set_title(f'{cfg["front_name"]}号码频率分布（近 {n} 期）',
                           fontsize=13, fontweight='bold')
        ax_front.grid(axis='y', alpha=0.2, linestyle='--')

        # 后区柱状图
        back_nums = list(range(back_min, back_max + 1))
        back_counts = [back_freq.get(n, 0) for n in back_nums]
        ax_back.bar(back_nums, back_counts,
                    color=[get_ball_color(n, cfg) for n in back_nums],
                    edgecolor='white', linewidth=0.8, alpha=0.85, zorder=3)
        for num, count in zip(back_nums, back_counts):
            if count > 0:
                ax_back.text(num, count + 0.1, str(count), ha='center', va='bottom',
                             fontsize=9, fontweight='bold', zorder=4)
        ax_back.set_xlim(back_min - 1, back_max + 1)
        ax_back.set_xticks(range(back_min, back_max + 1))
        ax_back.set_title(f'{cfg["back_name"]}号码频率（近 {n} 期）',
                          fontsize=13, fontweight='bold')
        ax_back.grid(axis='y', alpha=0.2, linestyle='--')

    else:
        # 无后区 (kl8)：80号码全频
        if lottery_type == 'kl8':
            freq = {}
            for _, row in data.iterrows():
                for num in row['num_list']:
                    num = int(num)
                    freq[num] = freq.get(num, 0) + 1

            nums_list = list(range(1, 81))
            counts = [freq.get(n, 0) for n in nums_list]

            fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#FAFAFA')
            ax.set_facecolor('#FAFAFA')

            ax.bar(nums_list, counts,
                   color=[get_ball_color(n, cfg) for n in nums_list],
                   edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
            for num, count in zip(nums_list, counts):
                if count > 0:
                    ax.text(num, count + 0.1, str(count), ha='center', va='bottom',
                            fontsize=4, fontweight='bold', zorder=4, alpha=0.8)

            ax.set_xlim(0.5, 80.5)
            ax.set_xticks(range(1, 81, 2))
            ax.set_xticklabels(range(1, 81, 2), fontsize=4, rotation=90)
            ax.set_title(f'{cfg["name"]} · 号码频率分布图（近 {n} 期 · 每期20码）',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('号码', fontsize=10)
            ax.set_ylabel('出现次数', fontsize=10)
            ax.grid(axis='y', alpha=0.2, linestyle='--')

            # 每10个号码区间标注均值
            for g in range(8):
                start = g * 10 + 1
                end = (g + 1) * 10
                avg = np.mean(counts[start - 1:end]) if counts[start - 1:end] else 0
                ax.text((start + end) / 2, max(counts) * 0.95 if max(counts) > 0 else 1,
                        f'均值:{avg:.1f}', ha='center', va='top', fontsize=6,
                        color='#666666', fontweight='bold')
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#FAFAFA')
            ax.text(0.5, 0.5, f'{cfg["name"]} 频率分布待实现',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cfg["name"]} 频率分布', fontsize=13)

    plt.tight_layout()
    return fig
