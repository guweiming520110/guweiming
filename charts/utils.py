#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
走势图绘图工具 — 基础绘图函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def draw_number_ball(ax, x, y, number, facecolor, edgecolor='white',
                     radius=0.38, fontsize=9, alpha=1.0, zorder=5):
    """绘制号码圆球"""
    circle = Circle((x, y), radius, facecolor=facecolor, edgecolor=edgecolor,
                    linewidth=1.5, alpha=alpha, zorder=zorder)
    ax.add_patch(circle)
    brightness = np.mean(facecolor[:3]) if hasattr(facecolor, '__getitem__') else 0.5
    text_color = 'white' if brightness < 0.55 else '#333333'
    ax.text(x, y, str(int(number)), ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder + 1)


def draw_connection(ax, x1, y1, x2, y2, color, alpha=0.35, linewidth=2.0, zorder=3):
    """绘制号码之间的连线"""
    ax.plot([x1, x2], [y1, y2], '-', color=color, alpha=alpha,
            linewidth=linewidth, solid_capstyle='round', zorder=zorder)


def get_ball_color(num, cfg):
    """历史号码颜色（统一绿色）"""
    return cfg['color_ball']


def should_connect(matrix, n, max_val, i, j, ni, nj, step=1):
    """
    判断两个号码是否应该连线（步长1=相邻, 步长2=中间空1格）

    规则几何图形:
      1. 直的线段   — 3+点共线（横/竖/对角）
      2. 方块       — 2×2 及以上矩形
      3. 三角形     — L型三点（直角三角）
      4. 十字形     — 中心点≥3个正交邻居
      5. T/Z型延伸  — 扩展图形
    """
    di, dj = ni - i, nj - j
    is_h_adj = (di == 0 and abs(dj) == 1)
    is_v_adj = (dj == 0 and abs(di) == 1)
    is_dr_adj = (di == 1 and dj == 1)
    is_dl_adj = (di == 1 and dj == -1)

    # ======== 步长=2：隔1空位规律检测 ========
    if step == 2:
        pattern_points = [1]
        for k in range(1, 4):
            fi, fj = i + di * k, j + dj * k
            if 0 <= fi < n and 1 <= fj <= max_val and matrix[fi][fj]:
                pattern_points.append(k)
        for k in range(-1, -4, -1):
            bi, bj = i + di * k, j + dj * k
            if 0 <= bi < n and 1 <= bj <= max_val and matrix[bi][bj]:
                pattern_points.insert(0, k)

        if len(pattern_points) >= 3:
            gaps = [pattern_points[t+1] - pattern_points[t] for t in range(len(pattern_points)-1)]
            if gaps and all(g % step == 0 for g in gaps):
                return True

        if is_h_adj or is_v_adj:
            for side_di, side_dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (di != 0 and side_di != 0) or (dj != 0 and side_dj != 0):
                    continue
                for ci, cj in [(i, j), (ni, nj)]:
                    ti, tj = ci + side_di, cj + side_dj
                    if 0 <= ti < n and 1 <= tj <= max_val and matrix[ti][tj]:
                        if not ((ti == i and tj == j) or (ti == ni and tj == nj)):
                            if abs(ti - i) + abs(tj - j) > 0 and abs(ti - ni) + abs(tj - nj) > 0:
                                return True

    # ======== step=1 的几何规则 ========
    if is_h_adj and ((j - 1 >= 1 and matrix[i][j - 1]) or (j + 2 <= max_val and matrix[i][j + 2])):
        return True
    if is_v_adj and ((i - 1 >= 0 and matrix[i - 1][j]) or (i + 2 < n and matrix[i + 2][j])):
        return True
    if is_dr_adj and ((i - 1 >= 0 and j - 1 >= 1 and matrix[i - 1][j - 1]) or
                      (i + 2 < n and j + 2 <= max_val and matrix[i + 2][j + 2])):
        return True
    if is_dl_adj and ((i - 1 >= 0 and j + 1 <= max_val and matrix[i - 1][j + 1]) or
                      (i + 2 < n and j - 2 >= 1 and matrix[i + 2][j - 2])):
        return True

    # 方块
    if is_h_adj:
        if (i - 1 >= 0 and matrix[i - 1][j] and matrix[i - 1][j + 1]):
            return True
        if (i + 1 < n and matrix[i + 1][j] and matrix[i + 1][j + 1]):
            return True
    if is_v_adj:
        if (j - 1 >= 1 and matrix[i][j - 1] and matrix[i + 1][j - 1]):
            return True
        if (j + 1 <= max_val and matrix[i][j + 1] and matrix[i + 1][j + 1]):
            return True

    # 三角形
    if is_h_adj:
        if (i + 1 < n and matrix[i + 1][j] and not matrix[i + 1][j + 1]):
            return True
        if (i + 1 < n and matrix[i + 1][j + 1] and not matrix[i + 1][j]):
            return True
        if (i - 1 >= 0 and matrix[i - 1][j] and not matrix[i - 1][j + 1]):
            return True
        if (i - 1 >= 0 and matrix[i - 1][j + 1] and not matrix[i - 1][j]):
            return True
    if is_v_adj:
        if (j + 1 <= max_val and matrix[i][j + 1] and not matrix[i + 1][j + 1]):
            return True
        if (j + 1 <= max_val and matrix[i + 1][j + 1] and not matrix[i][j + 1]):
            return True
        if (j - 1 >= 1 and matrix[i][j - 1] and not matrix[i + 1][j - 1]):
            return True
        if (j - 1 >= 1 and matrix[i + 1][j - 1] and not matrix[i][j - 1]):
            return True

    # 十字形
    def count_orthogonal(ci, cj):
        cnt = 0
        if ci - 1 >= 0 and matrix[ci - 1][cj]: cnt += 1
        if ci + 1 < n and matrix[ci + 1][cj]: cnt += 1
        if cj - 1 >= 1 and matrix[ci][cj - 1]: cnt += 1
        if cj + 1 <= max_val and matrix[ci][cj + 1]: cnt += 1
        return cnt

    if is_h_adj or is_v_adj:
        if count_orthogonal(i, j) >= 3 or count_orthogonal(ni, nj) >= 3:
            return True

    # T型
    if is_h_adj:
        if (j - 1 >= 1 and j + 2 <= max_val and matrix[i][j - 1] and matrix[i][j + 2]):
            if (i + 1 < n and matrix[i + 1][j]) or (i + 1 < n and matrix[i + 1][j + 1]):
                return True
            if (i - 1 >= 0 and matrix[i - 1][j]) or (i - 1 >= 0 and matrix[i - 1][j + 1]):
                return True
    if is_v_adj:
        if (i - 1 >= 0 and i + 2 < n and matrix[i - 1][j] and matrix[i + 2][j]):
            if (j + 1 <= max_val and matrix[i][j + 1]) or (j + 1 <= max_val and matrix[i + 1][j + 1]):
                return True
            if (j - 1 >= 1 and matrix[i][j - 1]) or (j - 1 >= 1 and matrix[i + 1][j - 1]):
                return True

    return False


def draw_cell_box(ax, row, col, face, edge, lw):
    """绘制一个网格单元格"""
    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                facecolor=face, edgecolor=edge,
                                linewidth=lw, zorder=1))


def get_reason_color(reason_label):
    """预测理由 → 颜色映射"""
    mapping = {
        '水平': '#4CAF50', '垂直': '#2196F3', '对角': '#FF9800',
        '三角': '#9C27B0', '方块': '#F44336', '十字': '#00BCD4',
        'T型': '#795548',
    }
    for kw, c in mapping.items():
        if kw in reason_label:
            return c
    return '#9E9E9E'
