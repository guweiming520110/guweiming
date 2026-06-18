#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心学习引擎
从历史矩阵中提取图案、统计频率、学习权重
"""

import numpy as np
from collections import defaultdict
from .matrix import LotteryMatrix
from .patterns import (
    PATTERN_TEMPLATES, find_all_patterns,
    count_patterns_around, get_pattern_label
)

# ==================== 可配置评分参数 ====================
# 所有数值均可通过回测自动优化
SCORE_PARAMS = {
    'rarity_min': 0.3,          # 稀有度下限
    'rarity_max': 5.0,          # 稀有度上限
    'span_bonus_factor': 0.5,   # 跨期连续因子
    'span_bonus_cap': 2.5,      # 跨期加成上限
    'horiz_penalty_max': 0.8,   # 水平惩罚系数 (0.8=纯水平×0.2, 同旧版)
    'diversity_factor': 0.05,   # 多样性奖励/种
    'omission_factor': 0.10,    # 遗漏值奖励系数
    'omission_max_mult': 2.0,   # 遗漏倍数上限
}


class PatternLearner:
    """几何图案学习引擎"""

    def __init__(self, lottery_type='dlt'):
        self.lottery_type = lottery_type
        self.matrix = None
        self.pattern_stats = {}   # {pattern_name: {count, freq, weight}}
        self.pattern_history = []  # 每期的图案统计

        if lottery_type == 'dlt':
            self.num_range = 35
            self.pick_count = 5
        elif lottery_type == 'dlt_back':
            self.num_range = 12
            self.pick_count = 2
        elif lottery_type == 'ssq':
            self.num_range = 33
            self.pick_count = 6
        elif lottery_type == 'ssq_back':
            self.num_range = 16
            self.pick_count = 1
        elif lottery_type == 'kl8':
            self.num_range = 80
            self.pick_count = 20
        else:
            raise ValueError(f"Unknown type: {lottery_type}")

    # ============ 训练 ============

    def train(self, df, num_periods=50):
        """
        训练学习器：从历史数据中提取图案并统计频率

        参数:
            df - DataFrame（大乐透或快乐8）
            num_periods - 使用的历史期数
        """
        print(f"\n{'='*50}")
        print(f"  图案学习引擎 - 训练开始")
        print(f"  彩种: {self.lottery_type.upper()}")
        print(f"{'='*50}")

        # 构建矩阵
        self.matrix = LotteryMatrix(self.lottery_type)
        if self.lottery_type == 'dlt':
            self.matrix.build_from_dlt(df, num_periods)
        elif self.lottery_type == 'dlt_back':
            self.matrix.build_from_dlt_back(df, num_periods)
        elif self.lottery_type == 'ssq':
            self.matrix.build_from_ssq(df, num_periods)
        elif self.lottery_type == 'ssq_back':
            self.matrix.build_from_ssq_back(df, num_periods)
        elif self.lottery_type == 'kl8':
            self.matrix.build_from_kl8(df, num_periods)

        print(f"  矩阵: {self.matrix.n_periods}期 × 1-{self.matrix.max_num}号")
        print(f"  图案类型: {len(PATTERN_TEMPLATES)}种")

        # 扫描全部图案
        all_patterns = find_all_patterns(self.matrix.matrix)

        total_patterns = sum(len(v) for v in all_patterns.values())
        print(f"  扫描到 {total_patterns} 个图案实例")

        # 统计频率
        self.pattern_stats = {}
        for p in PATTERN_TEMPLATES:
            name = p['name']
            count = len(all_patterns.get(name, []))
            freq = count / max(self.matrix.n_periods, 1)
            # 权重 = 频率归一化 (但至少0.1)
            weight = max(0.1, freq * 5)
            self.pattern_stats[name] = {
                'name': name,
                'label': p['label'],
                'count': count,
                'freq': freq,
                'weight': weight,
            }

        # 计算每期的图案参与度
        self._compute_period_patterns(all_patterns)

        self._print_stats()
        return self

    def _compute_period_patterns(self, all_patterns):
        """计算每期参与了多少图案"""
        n = self.matrix.n_periods
        period_scores = []

        for i in range(n):
            score = 0
            patterns_found = defaultdict(int)
            for pname, positions in all_patterns.items():
                # 统计该期参与了多少个此类型图案
                count = sum(1 for r, c in positions if r == i)
                if count > 0:
                    patterns_found[pname] = count
                    w = self.pattern_stats.get(pname, {}).get('weight', 1)
                    score += count * w
            period_scores.append({
                'period_idx': i,
                'period': self.matrix.periods[i] if self.matrix.periods is not None else str(i),
                'score': score,
                'patterns': dict(patterns_found),
            })

        self.pattern_history = period_scores

    def _print_stats(self):
        """打印统计信息"""
        print(f"\n  --- 图案频率统计 ---")
        sorted_stats = sorted(
            self.pattern_stats.values(),
            key=lambda x: x['count'], reverse=True
        )
        for s in sorted_stats[:10]:
            bar = '█' * min(int(s['freq'] * 30), 30)
            print(f"  {s['label']:12s} {bar} {s['count']:3d}次  freq={s['freq']:.2f}")

    # ============ 遗漏值计算 ============

    def _compute_omissions(self):
        """
        计算每个号码的遗漏期数（最近出现距今多少期）
        遗漏值越大表示号码越冷
        """
        if self.matrix is None:
            return {}

        n_periods = self.matrix.n_periods
        omissions = {}
        for num in range(1, self.num_range + 1):
            found = False
            for i in range(n_periods):
                if self.matrix.matrix[i][num]:
                    omissions[num] = i  # i=0 表示本期出现，遗漏0期
                    found = True
                    break
            if not found:
                omissions[num] = n_periods  # 从未出现 = 最大遗漏
        return omissions

    # ============ 评分 ============

    def score_candidates(self, candidates=None):
        """
        对候选号码进行优化评分（v2.0）

        评分规则优化:
        1. 图案复杂度：C=0.3-0.4, B=0.8-1.2, A=1.2-1.5, S=2.0-3.0 (base_score)
        2. 跨期加成：连续公式 1.0 + log(span+1)*factor
        3. 稀有度权重：log₂(总图案数/该频率) + 拉普拉斯平滑
        4. 水平惩罚：按比例动态惩罚，非纯水平也有轻微折扣
        5. 多样性奖励：增加至 5%/种
        6. 遗漏值奖励：冷号额外加分
        """
        if self.matrix is None:
            return []

        if candidates is None:
            candidates = list(range(1, self.num_range + 1))

        P = SCORE_PARAMS
        omissions = self._compute_omissions()

        # 从pattern_templates获取复杂度等级
        complexity_map = {}
        for p in PATTERN_TEMPLATES:
            complexity_map[p['name']] = {
                'base_score': p.get('base_score', 0.5),
                'complexity': p.get('complexity', 'C'),
            }

        results = []

        for num in candidates:
            test_matrix = self.matrix.add_candidate(num)
            if test_matrix is None:
                continue

            pattern_counts = count_patterns_around(
                test_matrix, row=0, col=num
            )

            if not pattern_counts:
                # 即使没有图案，遗漏值也可能加分
                omission_bonus = self._calc_omission_bonus(num, omissions, P)
                if omission_bonus > 0:
                    om_label = f'遗漏{omissions.get(num, 0):.0f}期'
                    results.append((num, round(omission_bonus, 2),
                                    {'遗漏': om_label}))
                else:
                    results.append((num, 0.0, {}))
                continue

            score = 0.0
            detail = {}
            total_patterns = sum(v.get('count', 0) for v in self.pattern_stats.values()
                                 if isinstance(v, dict)) or 1

            horiz_count = 0      # 单纯水平的图案数
            cross_period_count = 0  # 跨期图案数

            for pname, count in pattern_counts.items():
                info = complexity_map.get(pname, {'base_score': 0.5})
                base = info['base_score']

                # 图案定义查找
                pat_def = None
                for p in PATTERN_TEMPLATES:
                    if p['name'] == pname:
                        pat_def = p
                        break

                if pat_def:
                    pstep = pat_def.get('step', 1)
                    offsets = pat_def['offsets']
                    drs = [dr * pstep for dr, dc in offsets]
                    span = max(drs) - min(drs)
                    n_points = len(offsets)
                else:
                    span = 0
                    n_points = 3

                # --- 改进1: 稀有度权重 (拉普拉斯平滑 log₂) ---
                stat = self.pattern_stats.get(pname, {})
                freq = stat.get('count', 0)
                # log₂((total+1) / (freq+1)) → 避免0分母
                rarity = np.log2(max(2, total_patterns + 1) / max(freq + 1, 1))
                rarity = max(P['rarity_min'], min(P['rarity_max'], rarity))

                # --- 改进2: 跨期加成 (连续公式) ---
                if span > 0:
                    cross_period_count += count
                    span_bonus = min(P['span_bonus_cap'],
                                     1.0 + np.log(span + 1) * P['span_bonus_factor'])
                else:
                    span_bonus = 0.3  # 仅同期的图案价值低

                # 图案点数奖励
                point_bonus = 1.0 + (n_points - 3) * 0.2

                # 判断是否是纯水平图案
                is_horiz = (span == 0)
                if is_horiz:
                    horiz_count += count

                weighted = base * rarity * span_bonus * point_bonus * count
                score += weighted
                label = get_pattern_label(pname)
                detail[label] = {
                    'count': count,
                    'base_score': base,
                    'rarity': round(rarity, 2),
                    'span_bonus': span_bonus,
                    'span': span,
                    'weighted': round(weighted, 2),
                }

            # --- 改进3: 水平惩罚 (比例动态, 非全有全无) ---
            total_parts = sum(pattern_counts.values())
            if total_parts > 0:
                horiz_ratio = horiz_count / total_parts
                # horiz_ratio=1 → ×(1-0.8)=×0.2; horiz_ratio=0.5 → ×(1-0.4)=×0.6
                penalty = 1.0 - horiz_ratio * P['horiz_penalty_max']
                score *= penalty

            # --- 改进4: 多样性奖励 (5%/种) ---
            diversity = len(pattern_counts)
            score *= (1 + diversity * P['diversity_factor'])

            # --- 改进5: 遗漏值奖励 (冷号加分) ---
            omission_bonus = self._calc_omission_bonus(num, omissions, P)
            score += omission_bonus

            results.append((num, round(score, 2), detail))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _calc_omission_bonus(self, num, omissions, params):
        """计算遗漏值奖励分"""
        omission = omissions.get(num, 0)
        if omission <= 1:  # 最近刚出现过，不加分
            return 0.0
        # 遗漏越久加分越多，但上限封顶
        mult = min(omission * params['omission_factor'], params['omission_max_mult'])
        return round(mult, 2)

    # ============ 预测 ============

    def predict(self, top_k=None, exclude_recent=0):
        """
        预测下一期号码

        参数:
            top_k - 预测数量 (默认=该彩种每期开奖数×2)
            exclude_recent - 排除最近N期出现过的号码

        返回:
            list of (number, score, detail)
        """
        if top_k is None:
            top_k = self.pick_count * 2

        candidates = list(range(1, self.num_range + 1))

        # 排除最近出现的号码
        if exclude_recent > 0 and self.matrix is not None:
            recent = set()
            for i in range(min(exclude_recent, self.matrix.n_periods)):
                for n in range(1, self.num_range + 1):
                    if self.matrix.matrix[i][n]:
                        recent.add(n)
            candidates = [n for n in candidates if n not in recent]

        scored = self.score_candidates(candidates)
        return scored[:top_k]

    def predict_with_report(self, top_k=None):
        """
        预测并生成详细报告

        返回:
            predictions, summary_text
        """
        if top_k is None:
            top_k = self.pick_count * 2

        predictions = self.predict(top_k=top_k)

        lines = []
        lines.append(f"{'='*55}")
        lines.append(f"  几何图案预测报告")
        lines.append(f"  彩种: {self.lottery_type.upper()}")
        lines.append(f"{'='*55}")

        if not predictions:
            lines.append("  暂无预测结果")
            return [], '\n'.join(lines)

        lines.append(f"\n  Top-{len(predictions)} 预测号码:")
        lines.append(f"  {'排名':>4s} {'号码':>4s} {'评分':>8s}  {'参与图案'}")
        lines.append(f"  {'----':>4s} {'----':>4s} {'----':>8s}  {'--------'}")

        for rank, (num, score, detail) in enumerate(predictions, 1):
            # 汇总图案
            pattern_parts = []
            for label, d in sorted(detail.items(),
                                    key=lambda x: x[1]['weighted'] if isinstance(x[1], dict) and 'weighted' in x[1] else 0,
                                    reverse=True)[:5]:
                if isinstance(d, str):
                    pattern_parts.append(d)
                elif isinstance(d, dict) and 'count' in d:
                    pattern_parts.append(f"{label}x{d['count']}")
                else:
                    pattern_parts.append(str(d))
            pattern_str = '+'.join(pattern_parts) if pattern_parts else '-'

            bar = '█' * min(int(score), 20)
            lines.append(
                f"  #{rank:<2d}  {num:>3d}  {score:>6.1f}  {bar}  {pattern_str}"
            )

        # 统计信息
        total_patterns = sum(
            s['count'] for s in self.pattern_stats.values()
        )
        lines.append(f"\n  总图案数: {total_patterns}")
        lines.append(f"  覆盖期数: {self.matrix.n_periods}期")
        lines.append(f"{'='*55}")

        return predictions, '\n'.join(lines)

    # ============ 回测 ============

    def backtest(self, df, test_periods=20):
        """
        回测验证：逐期用历史数据预测，对比实际开奖

        参数:
            df - 完整历史数据
            test_periods - 回测期数

        返回:
            metrics dict
        """
        print(f"\n{'='*50}")
        print(f"  回测验证 - 最近 {test_periods} 期")
        print(f"{'='*50}")

        total = len(df)
        hit_counts = []
        top10_rates = []

        for t in range(test_periods):
            # 用前 total-test_periods+t 期数据训练
            train_df = df.iloc[:total - test_periods + t]
            if len(train_df) < 10:
                continue

            self.train(train_df, num_periods=min(50, len(train_df)))

            # 预测
            predictions = self.predict(top_k=self.pick_count * 2)
            predicted_nums = set(n for n, s, d in predictions[:self.pick_count])
            top10_nums = set(n for n, s, d in predictions[:10])

            # 实际开奖
            actual_row = df.iloc[total - test_periods + t]
            if self.lottery_type == 'kl8':
                actual = set(actual_row['num_list'])
            elif self.lottery_type in ('dlt', 'ssq'):
                front_cnt = 5 if self.lottery_type == 'dlt' else 6
                actual = set(int(actual_row[f'front_{k}']) for k in range(1, front_cnt + 1))
            elif self.lottery_type in ('dlt_back', 'ssq_back'):
                back_cnt = 2 if self.lottery_type == 'dlt_back' else 1
                actual = set(int(actual_row[f'back_{k}']) for k in range(1, back_cnt + 1))
            else:
                continue

            hit = len(predicted_nums & actual)
            hit_counts.append(hit)

            top10_hit = len(top10_nums & actual)
            top10_rates.append(top10_hit)

            period = actual_row.get('period', '?')
            print(f"  期{period}: 预测{list(predicted_nums)}... "
                  f"实开{sorted(actual)}... 命中{hit}/{self.pick_count}")

        if hit_counts:
            avg_hit = np.mean(hit_counts)
            avg_top10 = np.mean(top10_rates)
            print(f"\n  --- 回测结果 ---")
            print(f"  平均命中: {avg_hit:.2f}/{self.pick_count}")
            print(f"  平均Top10命中: {avg_top10:.2f}")
            print(f"  最大单期命中: {max(hit_counts)}")
            print(f"{'='*50}")

            return {
                'avg_hit': avg_hit,
                'avg_top10_hit': avg_top10,
                'max_hit': max(hit_counts),
                'hit_counts': hit_counts,
            }

        return None
