#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
号码连通网络图 — 配置驱动，支持有后区(dlt/ssq)和无后区(kl8)彩种
新增：支持显示历史各期预测号码对比
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from charts.utils import (
    draw_number_ball, draw_connection, should_connect,
    draw_cell_box, get_ball_color, get_reason_color
)
from charts.config import chart_config, zone_color_scheme, future_periods_count
from charts.config import HISTORY_BALL_COLOR


def load_history_predictions(lottery_type, db_path=None):
    """从数据库加载历史预测记录"""
    if db_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        db_path = base_dir / 'lottery.db'

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    predictions = {}
    try:
        cur.execute(
            """SELECT target_period, front_numbers, back_numbers,
                      actual_front, actual_back
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
        pass
    finally:
        conn.close()

    return predictions


def generate_network_chart(df, lottery_type='dlt', num_periods=30,
                           figsize=(24, 14),
                           predicted_front=None, predicted_back=None,
                           verified_front=None, verified_back=None,
                           predicted_reasons=None):
    """
    统一号码连通网络图

    布局：最旧期号(行0) → ... → 最新期号(行n-1)
          ──── 金色分隔线 ────
          未来+1(行n) → 未来+2(行n+1) → 未来+3(行n+2)

    对 has_back=True (dlt/ssq): 左前区 + gap + 右后区
    对 has_back=False (kl8):   仅前区，80列连续

    新增：历史各期预测号码显示
    - 黄色球: 预测命中
    - 红色球: 预测未命中
    - 黑色球: 待验证
    """
    cfg = chart_config(lottery_type)
    data = df.tail(num_periods).copy().reset_index(drop=True)
    n = len(data)
    if n == 0:
        return None

    has_back = cfg['has_back']
    front_min, front_max = cfg['front_min'], cfg['front_max']
    back_min, back_max = cfg['back_min'], cfg['back_max']
    gap_start, gap_end = cfg['gap_start'], cfg['gap_end']
    back_start, back_end = cfg['back_start'], cfg['back_end']
    back_shift_local = cfg['back_shift']

    front_cols = cfg['front_cols']
    back_cols = cfg['back_cols']

    # 未来期数
    future_p = future_periods_count()
    total_rows = n + future_p
    latest_period_num = int(data['period'].iloc[-1])
    future_nums = [str(latest_period_num + 1 + i) for i in range(future_p)]
    periods_labels = list(data['period'].astype(str).values) + future_nums

    # 加载历史预测记录
    history_preds = load_history_predictions(lottery_type)
    has_history_pred = len(history_preds) > 0

    # ---- 构建存在矩阵 ----
    front_matrix = np.zeros((total_rows, front_max + 1), dtype=bool)

    if has_back:
        back_matrix = np.zeros((total_rows, back_max + 1), dtype=bool)

    if lottery_type == 'kl8':
        # kl8: 从 num_list 读取
        for i in range(n):
            for num in data.iloc[i]['num_list']:
                front_matrix[i][int(num)] = True
    else:
        # dlt/ssq: 从 front_cols/back_cols 读取
        for i in range(n):
            if front_cols:
                for col in front_cols:
                    front_matrix[i][int(data.iloc[i][col])] = True
            if has_back and back_cols:
                for col in back_cols:
                    back_matrix[i][int(data.iloc[i][col])] = True

    # ======== 创建画布 ========
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    x_left = front_min - 1.5
    x_right = (back_end + 1.5) if has_back else (front_max + 1.5)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-1.5, total_rows - 0.5)

    # ======== 网格背景 ========
    def is_future_row(idx):
        return idx >= n

    # 前区网格
    for abs_idx in range(total_rows):
        is_future = is_future_row(abs_idx)
        for j in range(front_min, front_max + 1):
            if is_future:
                shade = '#FFF8E1' if (abs_idx + j) % 2 == 0 else '#FFFDF0'
                draw_cell_box(ax, abs_idx, j, shade, '#F0D090', 0.5)
            else:
                shade = '#F0F0F0' if (abs_idx + j) % 2 == 0 else '#F8F8F8'
                draw_cell_box(ax, abs_idx, j, shade, '#E6E6E6', 0.4)

    # gap 列
    if cfg['gap_cols'] > 0:
        for abs_idx in range(total_rows):
            is_future = is_future_row(abs_idx)
            for j in range(gap_start, gap_end + 1):
                if is_future:
                    draw_cell_box(ax, abs_idx, j, '#FFF0C0', '#F0D090', 0.5)
                else:
                    draw_cell_box(ax, abs_idx, j, '#EAEAEA', '#DDDDDD', 0.4)

    # 后区网格
    if has_back:
        for abs_idx in range(total_rows):
            is_future = is_future_row(abs_idx)
            for j in range(back_start, back_end + 1):
                orig = j - back_shift_local
                if is_future:
                    draw_cell_box(ax, abs_idx, j, '#FFF0E0', '#F0D090', 0.5)
                else:
                    shade = '#F0F5FF' if (abs_idx + orig) % 2 == 0 else '#F5F8FF'
                    draw_cell_box(ax, abs_idx, j, shade, '#DDE6F0', 0.4)

    # 前区颜色分区
    zone_info = zone_color_scheme(lottery_type)
    if zone_info:
        for zs, ze, zc, za in zone_info:
            ax.axvspan(zs - 0.5, ze + 0.5, facecolor=zc, alpha=za, zorder=0)
    else:
        # kl8 分区 (每10个号码一组)
        if lottery_type == 'kl8':
            kl8_colors = ['#FF6B6B', '#FFB347', '#FFD700', '#90EE90',
                          '#87CEEB', '#9370DB', '#FF69B4', '#20B2AA']
            for g in range(8):
                s = g * 10 + 1
                e = (g + 1) * 10
                ax.axvspan(s - 0.5, e + 0.5, facecolor=kl8_colors[g], alpha=0.025, zorder=0)
                ax.text((s + e) / 2, n + 0.2, f'{s}-{e}', ha='center', va='bottom',
                        fontsize=7, color=kl8_colors[g], fontweight='bold', alpha=0.5)
            for g in range(1, 8):
                ax.axvline(x=g * 10 + 0.5, color='#CCCCCC', linewidth=0.6, linestyle='--', zorder=2)

    # 后区浅色背景
    if has_back:
        ax.axvspan(back_start - 0.5, back_end + 0.5, facecolor='#2196F3', alpha=0.025, zorder=0)

    # 分隔线
    if cfg['gap_cols'] > 0:
        ax.axvline(x=gap_start - 0.5, color='#AAAAAA', linewidth=1.2, linestyle='-', zorder=2)

    # 金色分隔线 (最新历史期与未来期之间)
    if future_p > 0:
        ax.axhline(y=n - 0.5, color='#E8A020', linewidth=2.0, linestyle='-', zorder=2)

    # ======== 连线（历史期）========
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    connection_count = 0

    # 前区连线
    for i in range(n):
        for j in range(1, front_max + 1):
            if not front_matrix[i][j]:
                continue
            for step in [1, 2]:
                for dj, di in directions:
                    ni, nj = i + di * step, j + dj * step
                    if ni < 0 or ni >= n or nj < 1 or nj > front_max or not front_matrix[ni][nj]:
                        continue
                    if not should_connect(front_matrix, n, front_max, i, j, ni, nj, step):
                        continue
                    c1 = get_ball_color(j, cfg)
                    c2 = get_ball_color(nj, cfg)
                    avg_c = tuple((np.array(c1[:3]) + np.array(c2[:3])) / 2)
                    lw = 4.5 if step == 1 else 3.0
                    draw_connection(ax, j, i, nj, ni, color=avg_c, alpha=0.55, linewidth=lw, zorder=2)
                    connection_count += 1

    # 后区连线 (dlt/ssq)
    if has_back:
        for i in range(n):
            for bj in range(back_min, back_max + 1):
                if not back_matrix[i][bj]:
                    continue
                dj_disp = bj + back_shift_local
                for step in [1, 2]:
                    for dbj, di in directions:
                        ni, nbj = i + di * step, bj + dbj * step
                        if ni < 0 or ni >= n or nbj < back_min or nbj > back_max or not back_matrix[ni][nbj]:
                            continue
                        if not should_connect(back_matrix, n, back_max, i, bj, ni, nbj, step):
                            continue
                        ndj_disp = nbj + back_shift_local
                        c1 = get_ball_color(bj, cfg)
                        c2 = get_ball_color(nbj, cfg)
                        avg_c = tuple((np.array(c1[:3]) + np.array(c2[:3])) / 2)
                        lw = 4.5 if step == 1 else 3.0
                        draw_connection(ax, dj_disp, i, ndj_disp, ni, color=avg_c, alpha=0.55, linewidth=lw, zorder=2)
                        connection_count += 1

    # 将预测号码填入未来第1行（row_idx=n）
    has_pred = (predicted_front and len(predicted_front) > 0) or (predicted_back and len(predicted_back) > 0)
    if has_pred:
        if predicted_front:
            for num in predicted_front:
                if front_min <= num <= front_max:
                    front_matrix[n][num] = True
        if has_back and predicted_back:
            for num in predicted_back:
                if back_min <= num <= back_max:
                    back_matrix[n][num] = True

    # ---- 所有行连线（含历史+未来期）----
    for abs_idx in range(total_rows):
        # 前区
        for j in range(1, front_max + 1):
            if not front_matrix[abs_idx][j]:
                continue
            for step in [1, 2]:
                for dj, di in directions:
                    ni, nj = abs_idx + di * step, j + dj * step
                    if ni < 0 or ni >= total_rows or nj < 1 or nj > front_max or not front_matrix[ni][nj]:
                        continue
                    if not should_connect(front_matrix, total_rows, front_max, abs_idx, j, ni, nj, step):
                        continue
                    c1 = get_ball_color(j, cfg)
                    c2 = get_ball_color(nj, cfg)
                    avg_c = tuple((np.array(c1[:3]) + np.array(c2[:3])) / 2)
                    lw = 4.5 if step == 1 else 3.0
                    is_pred_line = (abs_idx >= n or ni >= n)
                    draw_connection(ax, j, abs_idx, nj, ni, color=avg_c,
                                    alpha=0.55 if not is_pred_line else 0.4,
                                    linewidth=lw if not is_pred_line else 3.0, zorder=2)
                    connection_count += 1

        # 后区
        if has_back:
            for bj in range(back_min, back_max + 1):
                if not back_matrix[abs_idx][bj]:
                    continue
                dj_disp = bj + back_shift_local
                for step in [1, 2]:
                    for dbj, di in directions:
                        ni, nbj = abs_idx + di * step, bj + dbj * step
                        if ni < 0 or ni >= total_rows or nbj < back_min or nbj > back_max or not back_matrix[ni][nbj]:
                            continue
                        if not should_connect(back_matrix, total_rows, back_max, abs_idx, bj, ni, nbj, step):
                            continue
                        ndj_disp = nbj + back_shift_local
                        c1 = get_ball_color(bj, cfg)
                        c2 = get_ball_color(nbj, cfg)
                        avg_c = tuple((np.array(c1[:3]) + np.array(c2[:3])) / 2)
                        is_pred_line = (abs_idx >= n or ni >= n)
                        lw = 4.5 if step == 1 else 3.0
                        draw_connection(ax, dj_disp, abs_idx, ndj_disp, ni, color=avg_c,
                                        alpha=0.55, linewidth=lw, zorder=2)
                        connection_count += 1

    # ---- 绘制历史号码圆球 ----
    for abs_idx in range(n):
        # 前区
        for j in range(1, front_max + 1):
            if front_matrix[abs_idx][j]:
                draw_number_ball(ax, j, abs_idx, j, get_ball_color(j, cfg), radius=0.38, fontsize=8, zorder=6)
        # 后区
        if has_back:
            for bj in range(back_min, back_max + 1):
                if back_matrix[abs_idx][bj]:
                    draw_number_ball(ax, bj + back_shift_local, abs_idx, bj, get_ball_color(bj, cfg),
                                     radius=0.38, fontsize=8, zorder=6)

    # ======== 绘制历史各期预测号码 ========
    if has_history_pred:
        for abs_idx in range(n):
            period = str(data.iloc[abs_idx]['period'])
            if period not in history_preds:
                continue

            pred = history_preds[period]
            pred_front = pred['front']
            pred_back = pred['back']
            verified = pred['verified']
            actual_front_set = pred['actual_front']
            actual_back_set = pred['actual_back']

            # 前区预测号码
            if pred_front:
                for num in pred_front:
                    if not (front_min <= num <= front_max):
                        continue
                    is_hit = verified and num in actual_front_set

                    if is_hit:
                        bg = (0.95, 0.8, 0.0)  # 亮黄色 - 命中
                        ec = (1.0, 1.0, 0.2)   # 黄色边框
                    elif verified:
                        bg = (0.75, 0.05, 0.05)  # 深红色 - 未命中
                        ec = (1.0, 0.2, 0.2)     # 红色边框
                    else:
                        bg = (0.12, 0.12, 0.12)  # 黑色 - 待验证
                        ec = (0.6, 0.6, 0.6)     # 灰色边框

                    # 预测球画在开奖球相同位置（无偏移），在上层显示，大小一致
                    draw_number_ball(ax, num, abs_idx, num, bg,
                                     edgecolor=ec, radius=0.38, fontsize=8, zorder=7)
                    # 命中加★标记
                    if is_hit:
                        ax.text(num, abs_idx, '★', ha='center', va='center',
                               fontsize=7, color='white', fontweight='bold', zorder=8)

            # 后区预测号码
            if has_back and pred_back:
                for num in pred_back:
                    if not (back_min <= num <= back_max):
                        continue
                    is_hit = verified and num in actual_back_set

                    if is_hit:
                        bg = (0.95, 0.8, 0.0)
                        ec = (1.0, 1.0, 0.2)
                    elif verified:
                        bg = (0.75, 0.05, 0.05)
                        ec = (1.0, 0.2, 0.2)
                    else:
                        bg = (0.12, 0.12, 0.12)
                        ec = (0.6, 0.6, 0.6)

                    # 预测球画在开奖球下方（向下偏移0.32）
                    draw_number_ball(ax, num + back_shift_local, abs_idx + 0.32, num, bg,
                                     edgecolor=ec, radius=0.30, fontsize=7, zorder=7)
                    if is_hit:
                        ax.text(num + back_shift_local, abs_idx + 0.32, '★', ha='center', va='center',
                               fontsize=7, color='white', fontweight='bold', zorder=8)

    # ---- 预测理由彩色连线 + 未来期预测号码标记 ----
    if has_pred:
        pred_row = n

        # 构建预测理由标签映射
        reason_map = {}
        if predicted_reasons:
            for num_str, info in predicted_reasons.items():
                num = int(num_str)
                pats = info.get('patterns', {})
                if pats:
                    top_pat = list(pats.keys())[0]
                    clr = get_reason_color(top_pat)
                    reason_map[num] = (top_pat, clr)

        # 前区预测理由连线
        if predicted_front and reason_map:
            for num in predicted_front:
                if num not in reason_map:
                    continue
                reason, clr = reason_map[num]
                for step in [1, 2]:
                    for dj, di in directions:
                        hi = pred_row - di * step
                        hj = num - dj * step
                        if hi < 0 or hj < 1 or hj > front_max:
                            continue
                        if not front_matrix[hi][hj]:
                            continue
                        if should_connect(front_matrix, total_rows, front_max, pred_row, num, hi, hj, step):
                            draw_connection(ax, num, pred_row, hj, hi, color=clr, alpha=0.85,
                                            linewidth=3.5, zorder=2)
                            connection_count += 1

        # 前区预测号码
        if predicted_front:
            for num in predicted_front:
                if not (front_min <= num <= front_max):
                    continue
                is_verified = verified_front and num in verified_front
                has_verification = verified_front is not None

                if is_verified:
                    bg = (0.9, 0.75, 0.1)
                    ec = (1.0, 0.85, 0.2)
                elif has_verification:
                    bg = (0.85, 0.1, 0.1)
                    ec = (1.0, 0.2, 0.2)
                else:
                    bg = (0.08, 0.08, 0.08)
                    ec = (0.5, 0.5, 0.5)

                draw_number_ball(ax, num, pred_row, num, bg,
                                 edgecolor=ec, radius=0.44, fontsize=10, zorder=10)
                ax.text(num, pred_row, str(num), ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white', zorder=11)
                if num in reason_map:
                    r, clr = reason_map[num]
                    ax.text(num, pred_row + 0.55, r, ha='center', va='top',
                            fontsize=5.5, fontweight='bold', color=clr, zorder=11)

        # 后区预测
        if has_back and predicted_back:
            for num in predicted_back:
                if not (back_min <= num <= back_max):
                    continue
                disp = num + back_shift_local
                is_verified = verified_back and num in verified_back
                has_verification = verified_back is not None

                if is_verified:
                    bg = (0.9, 0.75, 0.1)
                    ec = (1.0, 0.85, 0.2)
                elif has_verification:
                    bg = (0.85, 0.1, 0.1)
                    ec = (1.0, 0.2, 0.2)
                else:
                    bg = (0.08, 0.08, 0.08)
                    ec = (0.5, 0.5, 0.5)

                draw_number_ball(ax, disp, pred_row, num, bg,
                                 edgecolor=ec, radius=0.44, fontsize=10, zorder=10)
                ax.text(disp, pred_row, str(num), ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white', zorder=11)
                if num in reason_map:
                    r, clr = reason_map[num]
                    ax.text(disp, pred_row + 0.55, r, ha='center', va='top',
                            fontsize=5.5, fontweight='bold', color=clr, zorder=11)

    # ---- 期号标注 ----
    for i in range(n):
        r_label_x = (back_end + 1.2) if has_back else (front_max + 1.2)
        ax.text(r_label_x, i, periods_labels[i], fontsize=6, color='#999999', va='center', zorder=3)

    for fi in range(future_p):
        row = n + fi
        r_label_x = (back_end + 1.2) if has_back else (front_max + 1.2)
        clr = '#E8A020' if fi == 0 else '#CCCCCC'
        fw = 'bold' if fi == 0 else 'normal'
        fs = 7 if fi == 0 else 6
        ax.text(r_label_x, row, '→ ' + periods_labels[row],
                fontsize=fs, fontweight=fw, color=clr, va='center', zorder=3)

    # ---- 区域标题 ----
    front_mid = (front_min + front_max) / 2
    if has_back:
        ax.text(front_mid, total_rows - 0.5,
                f'{cfg["front_name"]}号码 ({front_min}-{front_max})',
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='#E74C3C', alpha=0.7)
        back_mid = (back_start + back_end) / 2
        ax.text(back_mid, total_rows - 0.5,
                f'{cfg["back_name"]} ({back_min}-{back_max})',
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='#2196F3', alpha=0.7)
    else:
        ax.text(front_mid, total_rows - 0.5,
                f'{cfg["front_name"]} ({front_min}-{front_max})',
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='#E74C3C', alpha=0.7)

    # ---- 坐标轴 ----
    front_ticks = list(range(front_min, front_max + 1, 2))
    if has_back:
        back_ticks = [back_start + i for i in range(back_min, back_max + 1)]
        all_ticks = front_ticks + back_ticks
        all_labels = [str(t) for t in front_ticks] + [str(t) for t in range(back_min, back_max + 1)]
    else:
        all_ticks = front_ticks
        all_labels = [str(t) for t in front_ticks]

    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=5.5)
    ax.set_yticks(range(total_rows))
    ax.set_yticklabels(periods_labels, fontsize=6)
    ax.invert_yaxis()
    ax.tick_params(axis='x', rotation=90, length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('号码', fontsize=11, labelpad=8)
    ax.set_ylabel('期数（从上到下）', fontsize=11, labelpad=8)

    # ---- 图例 ----
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=HISTORY_BALL_COLOR,
                   markersize=9, label='开奖号码(绿底白字)'),
        plt.Line2D([0], [0], color='#888888', linewidth=2, alpha=0.5, label='相邻号码连线'),
    ]
    if has_pred:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222',
                       markeredgecolor='white', markersize=9, label='预测号码(未验证)'))
        # 验证后的图例
        has_verified_any = (verified_front is not None) or (verified_back is not None)
        if has_verified_any:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EECC00',
                           markeredgecolor='#DDAA00', markersize=9, label='预测正确(黄底白字★)'))
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD2222',
                           markeredgecolor='#FF3333', markersize=9, label='预测错误(红底白字)'))
        for kw, clr, lbl in [('水平', '#4CAF50', '水平理由'), ('垂直', '#2196F3', '垂直理由'),
                              ('对角', '#FF9800', '对角理由'), ('三角', '#9C27B0', '三角理由')]:
            legend_elements.append(
                plt.Line2D([0], [0], color=clr, linewidth=3, alpha=0.8, label=lbl))
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.85, fontsize=8, ncol=6)

    # ---- 标题 ----
    first_p = periods_labels[0]
    last_hist = periods_labels[total_rows - future_p - 1]
    title_name = cfg['name']
    ax.set_title(f'{title_name} · 综合号码相邻连通网络图（{last_hist}~{first_p} · {connection_count}条连线）',
                 fontsize=14, fontweight='bold', pad=14)

    plt.tight_layout()
    return fig
