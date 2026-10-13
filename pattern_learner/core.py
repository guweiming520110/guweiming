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
    count_patterns_around, count_patterns_around_weighted,
    get_pattern_label
)

# ==================== 可配置评分参数 ====================
# 所有数值均可通过回测自动优化
SCORE_PARAMS = {
    'rarity_min': 0.3,          # 稀有度下限
    'rarity_max': 5.0,          # 稀有度上限
    'span_bonus_factor': 0.5,   # 跨期连续因子
    'span_bonus_cap': 2.5,      # 跨期加成上限
    'horiz_penalty_max': 0.4,   # 水平惩罚系数 (降低惩罚，原0.8)
    'diversity_factor': 0.10,   # 多样性奖励/种 (翻倍，原0.05)
    'omission_factor': 0.15,    # 遗漏值奖励系数 (提高，原0.10)
    'omission_max_mult': 3.5,   # 遗漏倍数上限 (提高，原2.0)
    'time_decay': 0.95,         # 时间衰减系数 (每期衰减5%)
    'time_decay_weight': 0.5,   # 时间衰减占评分的权重 (提高，原0.3)
    'hotness_window': 10,       # 近期热度窗口
    'hotness_bonus': 0.05,      # 冷号每期额外加成 (热度1/频率的奖励)
    'adaptive_window': False,   # 是否启用自适应窗口
    'window_max': 80,           # 自适应窗口最大期数
    'window_min': 20,           # 自适应窗口最小期数
    'last_n_penalty_factor': 0.8,  # 近N期号码降权系数 (新增)
    'last_n_penalty_count': 2,     # 对最近N期号码降权 (新增)
}


class PatternLearner:
    """几何图案学习引擎"""

    def __init__(self, lottery_type='dlt'):
        self.lottery_type = lottery_type
        # 初始化为空矩阵对象（而非 None）：build() 会重新构建并填充真实数据；
        # 这样所有 self.matrix.matrix / self.matrix.n_periods 访问在类型层面均安全，
        # 未训练时退化为 n_periods=0 的空矩阵，与调用方契约一致
        self.matrix: LotteryMatrix = LotteryMatrix(self.lottery_type)
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

        # ---- P3c: 自适应窗口 ----
        P = SCORE_PARAMS
        if P['adaptive_window']:
            if 'num_list' in df.columns:
                # KL8: 用号码数量密度估算熵。快乐8每期开20/80=0.25，
                # 密度越高代表越混乱（号码覆盖越广）→ 短窗口；密度低 → 长窗口。
                density = df['num_list'].apply(len).iloc[-30:].mean()
                # H1 修复: 原来 density 计算后从未使用，KL8 的 num_periods 恒不变，
                # 自适应窗口对 KL8 完全失效。现在按密度推导：
                # 每期开奖号码占比 0.1~0.3 之间归一化到 [0,1]，密度高(接近0.3)取短窗口。
                norm_density = min(max((density - 0.10) / 0.20, 0.0), 1.0) if density else 0.5
                window_range = P['window_max'] - P['window_min']
                num_periods = int(P['window_max'] - norm_density * window_range)
                print(f"  📐 自适应窗口(KL8): 密度={density:.1f} → {num_periods}期")
            else:
                # DLT/SSQ: 用号码分布的熵
                recent = df.tail(30)
                counts = np.zeros(self.num_range + 1)
                for _, row in recent.iterrows():
                    for k in range(1, self.pick_count + 1):
                        n = int(row.get(f'front_{k}', 0))
                        if 1 <= n <= self.num_range:
                            counts[n] += 1
                cnt = counts[1:]
                # 退化窗口(近30期无有效号码)时 cnt.sum()==0，else 分支不会赋值
                # entropy/max_entropy，而下方 print 会无条件引用它们 → 必须先给默认值
                entropy = 0.0
                max_entropy = 1.0
                if cnt.sum() > 0:
                    prob = cnt / cnt.sum()
                    entropy = -np.sum(prob * np.log2(prob + 1e-10))
                    # 归一化熵: 高熵(混乱) → 短窗口；低熵(稳定) → 长窗口
                    # SSQ/DLT: 理论最大熵~log2(35)≈5.13
                    max_entropy = np.log2(self.num_range)
                    norm_entropy = min(entropy / max_entropy, 1.0)
                else:
                    norm_entropy = 0.5
                # 高熵(=混乱)用短窗口，低熵(=稳定)用长窗口
                window_range = P['window_max'] - P['window_min']
                num_periods = int(P['window_max'] - norm_entropy * window_range)
                print(f"  📐 自适应窗口: 熵={entropy:.2f}/{max_entropy:.2f} → {num_periods}期")

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
        对候选号码进行优化评分（v3.0）

        评分规则优化:
        1. 图案复杂度：C=0.3-0.4, B=0.8-1.2, A=1.2-1.5, S=2.0-3.0 (base_score)
        2. 跨期加成：连续公式 1.0 + log(span+1)*factor
        3. 稀有度权重：log₂(总图案数/该频率) + 拉普拉斯平滑
        4. 水平惩罚：按比例动态惩罚
        5. 多样性奖励：5%/种
        6. 遗漏值奖励：冷号额外加分
        7. 时间衰减：近期图案权重更高 (P3a)
        8. 冷热号因子：近期频繁出现的号码做温和降权 (P3b)
        """
        if self.matrix is None:
            return []
        # C1 修复: 空矩阵（无训练数据）时无任何可评分图案，直接返回空。
        # 否则 add_candidate_inplace 的 matrix[0] 越界会抛 IndexError。
        if self.matrix.n_periods == 0:
            return []

        if candidates is None:
            candidates = list(range(1, self.num_range + 1))

        P = SCORE_PARAMS
        omissions = self._compute_omissions()
        hotness = self._compute_hotness(P['hotness_window'])

        # 从pattern_templates获取复杂度等级
        complexity_map = {}
        for p in PATTERN_TEMPLATES:
            complexity_map[p['name']] = {
                'base_score': p.get('base_score', 0.5),
                'complexity': p.get('complexity', 'C'),
            }

        results = []

        for num in candidates:
            # 性能优化：使用原地修改+恢复，避免每个候选号码都复制整个矩阵
            if not self.matrix.add_candidate_inplace(num):
                continue
            test_matrix = self.matrix.matrix  # 直接使用内部矩阵（已原地修改）

            # 使用时间衰减加权统计
            use_decay = P['time_decay_weight'] > 0
            # 始终初始化，避免 pyright 报 possibly-unbound；
            # 二者仅在各自分支被真实赋值，且后续均通过 `if use_decay` 三元选择访问，空字典不会被误用
            weighted_counts: dict = {}
            raw_counts: dict = {}
            if use_decay:
                weighted_counts = count_patterns_around_weighted(
                    test_matrix, row=0, col=num, decay=P['time_decay']
                )
            else:
                raw_counts = count_patterns_around(
                    test_matrix, row=0, col=num
                )

            if use_decay:
                has_pattern = any(v[0] > 0 for v in weighted_counts.values())
            else:
                has_pattern = bool(raw_counts)

            if not has_pattern:
                # 即使没有图案，遗漏值也可能加分
                omission_bonus = self._calc_omission_bonus(num, omissions, P)
                if omission_bonus > 0:
                    om_label = f'遗漏{omissions.get(num, 0):.0f}期'
                    results.append((num, round(omission_bonus, 2),
                                    {'遗漏': om_label}))
                else:
                    results.append((num, 0.0, {}))
                self.matrix.remove_candidate_inplace(num)
                continue

            score = 0.0
            detail = {}
            total_patterns = sum(v.get('count', 0) for v in self.pattern_stats.values()
                                 if isinstance(v, dict)) or 1

            horiz_count = 0
            cross_period_count = 0

            for pname, data in weighted_counts.items() if use_decay else raw_counts.items():
                if use_decay:
                    raw_count, decayed_count = data
                else:
                    raw_count = data
                    decayed_count = data

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

                # --- 改进1: 稀有度权重 ---
                stat = self.pattern_stats.get(pname, {})
                freq = stat.get('count', 0)
                rarity = np.log2(max(2, total_patterns + 1) / max(freq + 1, 1))
                rarity = max(P['rarity_min'], min(P['rarity_max'], rarity))

                # --- 改进2: 跨期加成 ---
                if span > 0:
                    cross_period_count += raw_count
                    span_bonus = min(P['span_bonus_cap'],
                                     1.0 + np.log(span + 1) * P['span_bonus_factor'])
                else:
                    span_bonus = 0.3

                point_bonus = 1.0 + (n_points - 3) * 0.2

                is_horiz = (span == 0)
                if is_horiz:
                    horiz_count += raw_count

                # --- P3a: 时间衰减权重 ---
                # 混合传统计数与衰减计数
                effective_count = (1 - P['time_decay_weight']) * raw_count + P['time_decay_weight'] * decayed_count

                weighted = base * rarity * span_bonus * point_bonus * effective_count
                score += weighted
                label = get_pattern_label(pname)
                detail[label] = {
                    'count': raw_count,
                    'decay_count': round(decayed_count, 2) if use_decay else raw_count,
                    'base_score': base,
                    'rarity': round(rarity, 2),
                    'span_bonus': span_bonus,
                    'span': span,
                    'weighted': round(weighted, 2),
                }

            # --- 改进3: 水平惩罚 ---
            total_parts = sum(
                v[0] if use_decay else v
                for v in (weighted_counts if use_decay else raw_counts).values()
            )
            if total_parts > 0:
                horiz_ratio = horiz_count / total_parts
                penalty = 1.0 - horiz_ratio * P['horiz_penalty_max']
                score *= penalty

            # --- 改进4: 多样性奖励 ---
            diversity = len(weighted_counts if use_decay else raw_counts)
            score *= (1 + diversity * P['diversity_factor'])

            # --- 改进5: 遗漏值奖励 ---
            omission_bonus = self._calc_omission_bonus(num, omissions, P)
            score += omission_bonus

            # --- P3b: 冷热号因子 (热号降权, 冷号升权) ---
            # 快乐8密度低，提高敏感度：超过平均密度就调整
            expected_density = self.pick_count / self.num_range  # 0.25 for kl8
            hot = hotness.get(num, 0)

            if hot > 0:
                # 热号降权：超过平均密度就降，降权幅度更大
                # 快乐8：近期出现3次以上就算热号
                if hot > expected_density * 1.2:
                    hot_penalty = 1.0 - (hot - expected_density * 1.2) * 0.8  # 最多降80%
                    hot_penalty = max(0.2, hot_penalty)  # 更激进的降权
                    score *= hot_penalty
            else:
                # 冷号升权：近期完全没出现过的号码升权
                cold_bonus = 1.0 + (expected_density * 0.5)  # 冷号+12.5%
                score *= cold_bonus

            # --- P3c: 最近N期开奖号码直接降权 (针对快乐8) ---
            # 最近几期刚开过的号码，再开概率相对低，直接降权
            if self.matrix is not None and self.matrix.n_periods > 0:
                last_n = P.get('last_n_penalty_count', 2)
                penalty_factor = P.get('last_n_penalty_factor', 0.8)
                recent_nums = set()
                max_i = min(last_n, self.matrix.n_periods)
                # 收集最近N期的所有开奖号码 (matrix[0]=最新)，越近惩罚越重
                for i in range(max_i):
                    for n in range(1, self.num_range + 1):
                        if self.matrix.matrix[i][n]:
                            recent_nums.add(n)
                            if num == n:
                                # 计算惩罚，越近惩罚越重
                                distance = i + 1  # i=0 → distance=1 = 最近一期
                                score *= penalty_factor ** distance

            results.append((num, round(score, 2), detail))
            self.matrix.remove_candidate_inplace(num)  # 恢复矩阵状态

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _calc_omission_bonus(self, num, omissions, params):
        """计算遗漏值奖励分"""
        omission = omissions.get(num, 0)
        if omission <= 1:  # 最近刚出现过，不加分
            return 0.0
        mult = min(omission * params['omission_factor'], params['omission_max_mult'])
        return round(mult, 2)

    def _compute_hotness(self, window=10):
        """
        计算近期每个号码的热度（最近 window 期出现频率）
        返回: {num: frequency}  0~1
        """
        if self.matrix is None or self.matrix.n_periods == 0:
            return {}

        n = min(window, self.matrix.n_periods)
        hotness = {}
        for num in range(1, self.num_range + 1):
            count = sum(1 for i in range(n) if self.matrix.matrix[i][num])
            hotness[num] = count / n if n > 0 else 0
        return hotness

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
                                    reverse=True):
                if isinstance(d, str):
                    pattern_parts.append(d)
                elif isinstance(d, dict) and d.get('count', 0) > 0:
                    pattern_parts.append(f"{label}x{d['count']}")
                elif isinstance(d, dict) and d.get('weighted', 0) > 0:
                    pattern_parts.append(label)
                if len(pattern_parts) >= 5:
                    break
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

        # 添加号码汇总
        top_nums = [n for n, s, d in predictions[:self.pick_count]]
        lines.append(f"\n  🎯 预测号码汇总: {sorted(top_nums)}")
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

            # M8 修复: 生成回测摘要文本，供 notifier 从报告中提取并拼入钉钉消息。
            # 旧实现回测只 print 到 stdout，predict_with_report 返回的报告不含回测段，
            # 导致 _extract_backtest_from_report 永远提取不到 → 钉钉消息缺回测统计。
            summary = (f"回测结果（最近{test_periods}期，{len(hit_counts)}期有效）\n"
                       f"  平均命中: {avg_hit:.2f}/{self.pick_count}\n"
                       f"  平均Top10命中: {avg_top10:.2f}\n"
                       f"  最大单期命中: {max(hit_counts)}")

            return {
                'avg_hit': avg_hit,
                'avg_top10_hit': avg_top10,
                'max_hit': max(hit_counts),
                'hit_counts': hit_counts,
                'summary': summary,
            }

        return None
