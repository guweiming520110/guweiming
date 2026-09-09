#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
号码趋势图 — 配置驱动，支持有后区(dlt/ssq)和无后区(kl8)彩种
新增：支持显示历史预测号码（同行为期对比）
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from charts.utils import draw_number_ball, get_ball_color
from charts.config import chart_config, zone_color_scheme


def load_history_predictions(lottery_type, db_path=None):
    """从数据库加载历史预测记录
    返回: {period: {'front': [nums], 'back': [nums], 'verified': bool}}
    """
    if db_path is None:
        # 默认数据库路径
        base_dir = Path(__file__).resolve().parent.parent.parent
        db_path = base_dir / 'lottery.db'

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    predictions = {}
    try:
        cur.execute(
            """SELECT target_period, front_numbers, back_numbers,
                      actual_front, actual_back, is_matched
               FROM prediction_records
               WHERE lottery_type = ?
               ORDER BY target_period ASC""",
            (lottery_type,)
        )

        for row in cur.fetchall():
            period = row[0]
            front_str = row[1]
            back_str = row[2]
            actual_front = row[3]
            actual_back = row[4]
            is_matched = row[5]

            front_nums = []
            if front_str:
                front_nums = [int(x.strip()) for x in front_str.split(',') if x.strip().isdigit()]

            back_nums = []
            if back_str:
                back_nums = [int(x.strip()) for x in back_str.split(',') if x.strip().isdigit()]

            verified = (actual_front is not None and actual_front != '')

            actual_front_set = set()
            actual_back_set = set()
            if verified and actual_front:
                actual_front_set = set(int(x.strip()) for x in actual_front.split(',') if x.strip().isdigit())
            if verified and actual_back:
                actual_back_set = set(int(x.strip()) for x in actual_back.split(',') if x.strip().isdigit())

            predictions[period] = {
                'front': front_nums,
                'back': back_nums,
                'verified': verified,
                'actual_front': actual_front_set,
                'actual_back': actual_back_set,
            }

    except Exception as e:
        print(f"加载历史预测记录失败: {e}")
    finally:
        conn.close()

    return predictions


def generate_trend_chart(df, lottery_type='dlt', num_periods=50, figsize=(24, 12),
                         predicted_front=None, predicted_back=None,
                         verified_front=None, verified_back=None):
    """统一号码趋势图
    同一期号同一位置：开奖号码与预测号码在相同x,y位置，仅以背景颜色区分
    绿色=开奖，黄色=预测命中，红色=预测未命中，黑色=待验证
    """
    cfg = chart_config(lottery_type)
    data = df.tail(num_periods).copy().reset_index(drop=True)
    n = len(data)
    if n == 0:
        return None

    periods = data['period'].astype(str).values
    has_back = cfg['has_back']
    front_min, front_max = cfg['front_min'], cfg['front_max']
    back_min, back_max = cfg['back_min'], cfg['back_max']
    front_cols = cfg['front_cols']
    back_cols = cfg['back_cols']

    # 加载历史预测记录
    history_preds = load_history_predictions(lottery_type)
    has_history_pred = len(history_preds) > 0

    if has_back:
        fig, (ax_front, ax_back) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1]),
            gridspec_kw={'height_ratios': [2.5, 1.2]},
            facecolor='#FAFAFA'
        )
        ax_front.set_facecolor('#FAFAFA')
        ax_back.set_facecolor('#FAFAFA')

        # ======== 前区 ========
        # 绘制开奖号码之间的连线（淡红色）
        for i in range(n):
            nums = sorted([int(data.iloc[i][col]) for col in front_cols])
            ax_front.plot([i] * len(nums), nums, '-', color='#E74C3C', alpha=0.15, linewidth=2.0, zorder=2)

        # 绘制开奖号码球（绿色背景，底层）
        for col in front_cols:
            for i in range(n):
                num = int(data.iloc[i][col])
                draw_number_ball(ax_front, i, num, num, get_ball_color(num, cfg),
                                 radius=0.38, fontsize=8, alpha=0.95, zorder=5)

        # ======== 绘制前区预测号码（同一位置，不同颜色，上层）========
        if has_history_pred:
            for i in range(n):
                period = str(data.iloc[i]['period'])
                if period not in history_preds:
                    continue

                pred = history_preds[period]
                pred_front = pred['front']
                verified = pred['verified']
                actual_set = pred['actual_front']

                if not pred_front:
                    continue

                # 预测号码球画在同一个位置，稍小一点，带边框
                for num in sorted(pred_front):
                    if not (front_min <= num <= front_max):
                        continue

                    is_hit = verified and num in actual_set

                    if is_hit:
                        bg = (0.95, 0.8, 0.0)  # 亮黄色 - 命中
                        ec = (1.0, 1.0, 0.2)   # 黄色边框
                    elif verified:
                        bg = (0.75, 0.05, 0.05)  # 深红色 - 未命中
                        ec = (1.0, 0.2, 0.2)     # 红色边框
                    else:
                        bg = (0.12, 0.12, 0.12)  # 黑色 - 待验证
                        ec = (0.6, 0.6, 0.6)     # 灰色边框

                    # 预测球稍小一点，在开奖球的上层
                    draw_number_ball(ax_front, i, num, num, bg,
                                     edgecolor=ec, radius=0.28, fontsize=7, alpha=0.95, zorder=7)

                    # 如果命中，在球中心加一个小星标记
                    if is_hit:
                        ax_front.text(i, num, '★', ha='center', va='center',
                                      fontsize=5, color='white', fontweight='bold', zorder=8)

        # 颜色分区
        zone_info = zone_color_scheme(lottery_type)
        if zone_info:
            for zs, ze, zc, za in zone_info:
                ax_front.axhspan(zs - 0.5, ze + 0.5, facecolor=zc, alpha=za, zorder=0)

        ax_front.set_ylim(front_min - 1, front_max + 2)
        ax_front.set_xlim(-0.8, n - 1 + 0.8)
        ax_front.set_ylabel(f'{cfg["front_name"]}号码', fontsize=11)
        ax_front.set_yticks(range(front_min, front_max + 1, 2))
        front_tick_labels = [str(t) for t in range(front_min, front_max + 1, 2)]
        ax_front.set_yticklabels(front_tick_labels, fontsize=7)
        ax_front.set_title(f'{cfg["name"]} · 号码趋势图（{cfg["front_name"]}+{cfg["back_name"]}综合）',
                           fontsize=14, fontweight='bold', pad=10)
        ax_front.grid(True, alpha=0.12, linestyle='--')

        # 图例说明
        if has_history_pred:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
                           markersize=10, label='开奖号码'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1C40F',
                           markeredgecolor='#FFFF33', markersize=8, label='预测命中 ★'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
                           markeredgecolor='#FF5555', markersize=8, label='预测未命中'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
                           markeredgecolor='#999999', markersize=8, label='待验证'),
            ]
            ax_front.legend(handles=legend_elements, loc='upper right',
                           framealpha=0.9, fontsize=9, ncol=4)

        # ======== 后区 ========
        # 绘制开奖号码之间的连线
        for i in range(n):
            nums = sorted([int(data.iloc[i][col]) for col in back_cols])
            ax_back.plot([i] * len(nums), nums, '-', color='#3498DB', alpha=0.25, linewidth=2.5, zorder=2)

        # 绘制开奖号码球（绿色背景）
        for col in back_cols:
            for i in range(n):
                num = int(data.iloc[i][col])
                draw_number_ball(ax_back, i, num, num, get_ball_color(num, cfg),
                                 radius=0.42, fontsize=10, zorder=5)

        # ======== 绘制后区预测号码（同一位置）========
        if has_history_pred:
            for i in range(n):
                period = str(data.iloc[i]['period'])
                if period not in history_preds:
                    continue

                pred = history_preds[period]
                pred_back = pred['back']
                verified = pred['verified']
                actual_set = pred['actual_back']

                if not pred_back:
                    continue

                for num in sorted(pred_back):
                    if not (back_min <= num <= back_max):
                        continue

                    is_hit = verified and num in actual_set

                    if is_hit:
                        bg = (0.95, 0.8, 0.0)
                        ec = (1.0, 1.0, 0.2)
                    elif verified:
                        bg = (0.75, 0.05, 0.05)
                        ec = (1.0, 0.2, 0.2)
                    else:
                        bg = (0.12, 0.12, 0.12)
                        ec = (0.6, 0.6, 0.6)

                    draw_number_ball(ax_back, i, num, num, bg,
                                     edgecolor=ec, radius=0.32, fontsize=9, alpha=0.95, zorder=7)

                    if is_hit:
                        ax_back.text(i, num, '★', ha='center', va='center',
                                    fontsize=6, color='white', fontweight='bold', zorder=8)

        ax_back.set_ylim(back_min - 0.5, back_max + 1.5)
        ax_back.set_xlim(-0.8, n - 1 + 0.8)
        ax_back.set_xlabel('开奖期号', fontsize=11)
        ax_back.set_ylabel(f'{cfg["back_name"]}号码', fontsize=11)
        ax_back.set_yticks(range(back_min, back_max + 1))
        back_tick_labels = [str(t) for t in range(back_min, back_max + 1)]
        ax_back.set_yticklabels(back_tick_labels, fontsize=8)

        tick_step = max(1, n // 14)
        for ax_tmp in [ax_front, ax_back]:
            ax_tmp.set_xticks(range(0, n, tick_step))
            ax_tmp.set_xticklabels(periods[::tick_step], rotation=45, ha='right', fontsize=7)
        ax_back.grid(True, alpha=0.12, linestyle='--')

    else:
        # 无后区 (kl8)
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0], 12), facecolor='#FAFAFA')
        ax.set_facecolor('#FAFAFA')

        # 绘制连线
        for i in range(n):
            num_list = sorted(data.iloc[i]['num_list'])
            ax.plot([i] * len(num_list), num_list, '-', color='#E74C3C', alpha=0.08, linewidth=1.5, zorder=2)

        # 绘制开奖号码球
        for i in range(n):
            for num in sorted(data.iloc[i]['num_list']):
                draw_number_ball(ax, i, num, num, get_ball_color(num, cfg),
                                 radius=0.28, fontsize=5, alpha=0.85, zorder=5)

        # ======== 绘制kl8预测号码（同一位置）========
        if has_history_pred:
            for i in range(n):
                period = str(data.iloc[i]['period'])
                if period not in history_preds:
                    continue

                pred = history_preds[period]
                pred_front = pred['front']
                verified = pred['verified']
                actual_set = pred['actual_front']

                if not pred_front:
                    continue

                for num in sorted(pred_front):
                    if not (front_min <= num <= front_max):
                        continue

                    is_hit = verified and num in actual_set

                    if is_hit:
                        bg = (0.95, 0.8, 0.0)
                        ec = (1.0, 1.0, 0.2)
                    elif verified:
                        bg = (0.75, 0.05, 0.05)
                        ec = (1.0, 0.2, 0.2)
                    else:
                        bg = (0.12, 0.12, 0.12)
                        ec = (0.6, 0.6, 0.6)

                    draw_number_ball(ax, i, num, num, bg,
                                     edgecolor=ec, radius=0.20, fontsize=4, alpha=0.9, zorder=7)

                    if is_hit:
                        ax.text(i, num, '★', ha='center', va='center',
                               fontsize=3, color='white', fontweight='bold', zorder=8)

        # 8区背景色
        kl8_colors = ['#FF6B6B', '#FFB347', '#FFD700', '#90EE90',
                      '#87CEEB', '#9370DB', '#FF69B4', '#20B2AA']
        for g in range(8):
            s = g * 10 + 1
            e = (g + 1) * 10
            ax.axhspan(s - 0.5, e + 0.5, facecolor=kl8_colors[g], alpha=0.03, zorder=0)

        ax.set_ylim(front_min - 1, front_max + 2)
        ax.set_xlim(-0.8, n - 1 + 0.8)
        ax.set_ylabel('号码', fontsize=11)
        ax.set_yticks(range(front_min, front_max + 1, 5))
        ax.set_yticklabels([str(t) for t in range(front_min, front_max + 1, 5)], fontsize=6)
        ax.set_title(f'{cfg["name"]} · 号码趋势图', fontsize=14, fontweight='bold', pad=10)

        tick_step = max(1, n // 14)
        ax.set_xticks(range(0, n, tick_step))
        ax.set_xticklabels(periods[::tick_step], rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.12, linestyle='--')

        # 图例
        if has_history_pred:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
                           markersize=8, label='开奖'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1C40F',
                           markeredgecolor='#FFFF33', markersize=6, label='命中★'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
                           markeredgecolor='#FF5555', markersize=6, label='未命中'),
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                     framealpha=0.9, fontsize=7, ncol=3)

    plt.tight_layout()
    return fig
