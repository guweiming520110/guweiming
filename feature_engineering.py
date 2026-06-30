import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from scipy.stats import entropy, skew, kurtosis
from sklearn.neighbors import NearestNeighbors


class FeatureEngineer:
    def __init__(self, df, lottery_type='dlt'):
        """
        初始化特征工程器

        参数:
            df: 历史数据DataFrame
            lottery_type: 彩票类型 (dlt=大乐透, ssq=双色球, kl8=快乐8, fc3d=福彩3D)
        """
        self.df = df.copy()
        self.lottery_type = lottery_type

        # 根据彩票类型设置号码范围和列名
        if lottery_type == 'dlt':
            self.front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
            self.back_cols = ['back_1', 'back_2']
            self.front_range = (1, 36)
            self.back_range = (1, 13)
        elif lottery_type == 'ssq':
            self.front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5', 'front_6']
            self.back_cols = ['back_1']
            self.front_range = (1, 34)
            self.back_range = (1, 17)
        elif lottery_type == 'kl8':
            self.front_cols = []  # kl8 uses 'num_list' column
            self.back_cols = []
            self.front_range = (1, 81)
            self.back_range = None
        elif lottery_type == 'fc3d':
            self.front_cols = ['front_1', 'front_2', 'front_3']
            self.back_cols = []
            self.front_range = (0, 10)
            self.back_range = None
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def extract_all_features(self):
        self.df = self.df.sort_values('period').reset_index(drop=True)

        self.add_number_frequency()
        self.add_omission_values()
        self.add_odd_even_ratio()
        self.add_size_ratio()
        self.add_sum_values()
        self.add_ac_value()
        self.add_span_value()

        self.add_cooccurrence_features()
        self.add_temporal_features()
        self.consecutive_features()
        self.add_interval_density()
        self.add_volatility_features()
        self.add_similarity_features()

        self.add_short_term_momentum()
        self.add_conditional_gap_features()
        self.add_transition_features()

        # === 新增高阶特征 ===
        self.add_omission_distribution_features()
        self.add_tail_features()
        self.add_sum_trend_features()
        self.add_ac_span_trend_features()
        self.add_edge_code_features()
        self.add_repeat_number_features()

        return self.df

    def add_number_frequency(self):
        front_freq = {}
        back_freq = {}

        # 初始化频率字典
        for num in range(self.front_range[0], self.front_range[1]):
            front_freq[num] = 0
        if self.back_range:
            for num in range(self.back_range[0], self.back_range[1]):
                back_freq[num] = 0

        # 统计频率
        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                for num in nums:
                    if self.front_range[0] <= num < self.front_range[1]:
                        front_freq[num] = front_freq.get(num, 0) + 1
            else:
                for col in self.front_cols:
                    if pd.notna(row[col]):
                        front_freq[int(row[col])] = front_freq.get(int(row[col]), 0) + 1
                for col in self.back_cols:
                    if pd.notna(row[col]):
                        back_freq[int(row[col])] = back_freq.get(int(row[col]), 0) + 1

        self.front_freq = front_freq
        self.back_freq = back_freq

    def add_omission_values(self):
        # 初始化遗漏值字典
        front_omission = {i: 0 for i in range(self.front_range[0], self.front_range[1])}
        back_omission = {}
        if self.back_range:
            back_omission = {i: 0 for i in range(self.back_range[0], self.back_range[1])}

        omission_records = []

        for idx, row in self.df.iterrows():
            # 获取当前期号码
            front_nums = []
            back_nums = []
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    front_nums = [int(x) for x in nums.split(',') if x.strip()]
            else:
                front_nums = [row[col] for col in self.front_cols]
                back_nums = [row[col] for col in self.back_cols] if self.back_cols else []

            for num in range(self.front_range[0], self.front_range[1]):
                if num in front_nums:
                    front_omission[num] = 0
                else:
                    front_omission[num] += 1

            if self.back_range:
                for num in range(self.back_range[0], self.back_range[1]):
                    if num in back_nums:
                        back_omission[num] = 0
                    else:
                        back_omission[num] += 1

            omission_records.append({
                'max_front_omission': max(front_omission.values()) if front_omission else 0,
                'avg_front_omission': np.mean(list(front_omission.values())) if front_omission else 0,
                'min_front_omission': min(front_omission.values()) if front_omission else 0,
                'max_back_omission': max(back_omission.values()) if back_omission else 0,
                'avg_back_omission': np.mean(list(back_omission.values())) if back_omission else 0,
            })

        for key in omission_records[0].keys():
            self.df[key] = [r[key] for r in omission_records]

    def add_odd_even_ratio(self):
        odd_counts = []
        even_counts = []

        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                odd = sum(1 for num in nums if num % 2 == 1)
                even = len(nums) - odd
            else:
                odd = sum(1 for col in self.front_cols if row[col] % 2 == 1)
                even = len(self.front_cols) - odd
            odd_counts.append(odd)
            even_counts.append(even)

        self.df['odd_count'] = odd_counts
        self.df['even_count'] = even_counts

    def add_size_ratio(self):
        # 动态计算大小号阈值（取号码范围的中点）
        threshold = (self.front_range[0] + self.front_range[1]) // 2

        large_counts = []
        small_counts = []

        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                large = sum(1 for num in nums if num > threshold)
                small = len(nums) - large
            else:
                large = sum(1 for col in self.front_cols if row[col] > threshold)
                small = len(self.front_cols) - large
            large_counts.append(large)
            small_counts.append(small)

        self.df['large_count'] = large_counts
        self.df['small_count'] = small_counts

    def add_sum_values(self):
        sums = []
        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                total = sum(nums)
            else:
                total = sum(row[col] for col in self.front_cols)
            sums.append(total)

        self.df['front_sum'] = sums

    def add_ac_value(self):
        # AC值 = 差值个数 - (号码个数-1)
        if self.lottery_type == 'kl8':
            # 快乐8选20，不计算AC值
            return

        ac_offset = len(self.front_cols) - 1
        ac_values = []
        for _, row in self.df.iterrows():
            nums = sorted([row[col] for col in self.front_cols])
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(abs(nums[i] - nums[j]))
            ac = len(diffs) - ac_offset
            ac_values.append(ac)

        self.df['ac_value'] = ac_values

    def add_span_value(self):
        spans = []
        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                span = max(nums) - min(nums)
            else:
                nums = [row[col] for col in self.front_cols]
                span = max(nums) - min(nums)
            spans.append(span)

        self.df['span'] = spans

    def get_hot_cold_numbers(self, top_n=10):
        sorted_front = sorted(self.front_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_back = sorted(self.back_freq.items(), key=lambda x: x[1], reverse=True)

        hot_front = sorted_front[:top_n]
        cold_front = sorted_front[-top_n:]
        hot_back = sorted_back[:5]
        cold_back = sorted_back[-5:]

        return {
            'hot_front': hot_front,
            'cold_front': cold_front,
            'hot_back': hot_back,
            'cold_back': cold_back
        }

    def add_cooccurrence_features(self):
        pair_counts = Counter()
        triplet_counts = Counter()
        total_draws = len(self.df)

        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                nums = sorted(nums)
            else:
                nums = sorted([row[col] for col in self.front_cols])

            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair = (nums[i], nums[j])
                    pair_counts[pair] += 1

            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    for k in range(j+1, len(nums)):
                        triplet = (nums[i], nums[j], nums[k])
                        triplet_counts[triplet] += 1

        pair_scores = []
        triplet_scores = []
        pmi_scores = []

        for idx, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                nums = sorted(nums)
            else:
                nums = sorted([row[col] for col in self.front_cols])

            max_pair_freq = 0
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair = (nums[i], nums[j])
                    if pair in pair_counts:
                        max_pair_freq = max(max_pair_freq, pair_counts[pair])

            max_triplet_freq = 0
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    for k in range(j+1, len(nums)):
                        triplet = (nums[i], nums[j], nums[k])
                        if triplet in triplet_counts:
                            max_triplet_freq = max(max_triplet_freq, triplet_counts[triplet])

            max_pmi = 0
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair = (nums[i], nums[j])
                    if pair in pair_counts:
                        p_xy = pair_counts[pair] / total_draws
                        p_x = self.front_freq.get(nums[i], 0) / (total_draws * len(self.front_cols)) if self.front_cols else self.front_freq.get(nums[i], 0) / (total_draws * 20)
                        p_y = self.front_freq.get(nums[j], 0) / (total_draws * len(self.front_cols)) if self.front_cols else self.front_freq.get(nums[j], 0) / (total_draws * 20)

                        if p_xy > 0 and p_x > 0 and p_y > 0:
                            pmi = np.log2(p_xy / (p_x * p_y))
                            max_pmi = max(max_pmi, pmi)

            pmi = max_pmi

            pair_scores.append(max_pair_freq)
            triplet_scores.append(max_triplet_freq)
            pmi_scores.append(pmi)

        self.df['max_pair_freq'] = pair_scores
        self.df['max_triplet_freq'] = triplet_scores
        self.df['pmi_score'] = pmi_scores

    def add_temporal_features(self):
        date_col = 'date' if 'date' in self.df.columns else 'draw_date'
        if date_col not in self.df.columns:
            return

        weekdays = []
        months = []
        weekday_sin = []
        weekday_cos = []
        month_sin = []
        month_cos = []

        for _, row in self.df.iterrows():
            try:
                if pd.isna(row[date_col]):
                    weekday = 0
                    month = 1
                else:
                    date_obj = pd.to_datetime(row[date_col])
                    weekday = date_obj.weekday()
                    month = date_obj.month
            except:
                weekday = 0
                month = 1

            weekdays.append(weekday)
            months.append(month)

            weekday_sin.append(np.sin(2 * np.pi * weekday / 7))
            weekday_cos.append(np.cos(2 * np.pi * weekday / 7))
            month_sin.append(np.sin(2 * np.pi * month / 12))
            month_cos.append(np.cos(2 * np.pi * month / 12))

        self.df['weekday'] = weekdays
        self.df['month'] = months
        self.df['weekday_sin'] = weekday_sin
        self.df['weekday_cos'] = weekday_cos
        self.df['month_sin'] = month_sin
        self.df['month_cos'] = month_cos

    def consecutive_features(self):
        consecutive_counts = []
        max_consecutive = []

        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                nums = sorted(nums)
            else:
                nums = sorted([row[col] for col in self.front_cols])

            consec_count = 0
            max_consec = 0
            current_consec = 1

            for i in range(len(nums)-1):
                if nums[i+1] - nums[i] == 1:
                    current_consec += 1
                    consec_count += 1
                else:
                    max_consec = max(max_consec, current_consec)
                    current_consec = 1

            max_consec = max(max_consec, current_consec)

            consecutive_counts.append(consec_count)
            max_consecutive.append(max_consec)

        self.df['consecutive_count'] = consecutive_counts
        self.df['max_consecutive'] = max_consecutive

    def add_interval_density(self):
        # 分三区
        if self.lottery_type == 'dlt':
            # 1-12, 13-24, 25-35
            zone1_counts = []
            zone2_counts = []
            zone3_counts = []
            for _, row in self.df.iterrows():
                nums = [row[col] for col in self.front_cols]
                zone1 = sum(1 for n in nums if 1 <= n <= 12)
                zone2 = sum(1 for n in nums if 13 <= n <= 24)
                zone3 = sum(1 for n in nums if 25 <= n <= 35)
                zone1_counts.append(zone1)
                zone2_counts.append(zone2)
                zone3_counts.append(zone3)
        elif self.lottery_type == 'ssq':
            # 1-11, 12-22, 23-33
            zone1_counts = []
            zone2_counts = []
            zone3_counts = []
            for _, row in self.df.iterrows():
                nums = [row[col] for col in self.front_cols]
                zone1 = sum(1 for n in nums if 1 <= n <= 11)
                zone2 = sum(1 for n in nums if 12 <= n <= 22)
                zone3 = sum(1 for n in nums if 23 <= n <= 33)
                zone1_counts.append(zone1)
                zone2_counts.append(zone2)
                zone3_counts.append(zone3)
        elif self.lottery_type == 'kl8':
            # 快乐8分10区
            zone_counts = {i: [] for i in range(1, 11)}
            zone_size = 8
            for _, row in self.df.iterrows():
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
                for i in range(1, 11):
                    start = (i-1) * zone_size + 1
                    end = i * zone_size
                    cnt = sum(1 for n in nums if start <= n <= end)
                    zone_counts[i].append(cnt)
            for i in range(1, 11):
                self.df[f'zone_{i}_count'] = zone_counts[i]
            return
        else:
            zone1_counts = []
            zone2_counts = []
            zone3_counts = []
            for _, row in self.df.iterrows():
                nums = [row[col] for col in self.front_cols]
                zone1 = sum(1 for n in nums if 1 <= n <= 12)
                zone2 = sum(1 for n in nums if 13 <= n <= 24)
                zone3 = sum(1 for n in nums if 25 <= n <= 35)
                zone1_counts.append(zone1)
                zone2_counts.append(zone2)
                zone3_counts.append(zone3)

        self.df['zone1_count'] = zone1_counts
        self.df['zone2_count'] = zone2_counts
        self.df['zone3_count'] = zone3_counts

    def add_volatility_features(self, window=10):
        rolling_std = []
        rolling_entropy = []

        for idx in range(len(self.df)):
            start_idx = max(0, idx - window + 1)
            window_data = self.df.iloc[start_idx:idx+1]

            all_nums = []
            for _, row in window_data.iterrows():
                if self.lottery_type == 'kl8':
                    nums = row.get('num_list', [])
                    if isinstance(nums, str):
                        nums = [int(x) for x in nums.split(',') if x.strip()]
                    all_nums.extend(nums)
                else:
                    all_nums.extend([row[col] for col in self.front_cols])

            if len(all_nums) > 0:
                std_val = np.std(all_nums)

                num_counts = np.bincount(all_nums, minlength=self.front_range[1])[self.front_range[0]:self.front_range[1]]
                probs = num_counts / (num_counts.sum() + 1e-10)
                entropy_val = entropy(probs + 1e-10)
            else:
                std_val = 0
                entropy_val = 0

            rolling_std.append(std_val)
            rolling_entropy.append(entropy_val)

        self.df['rolling_std'] = rolling_std
        self.df['rolling_entropy'] = rolling_entropy

    def add_similarity_features(self, k=5):
        if self.lottery_type == 'kl8':
            # KL8号码数量不一致，跳过
            return

        feature_matrix = self.df[self.front_cols].values

        if len(feature_matrix) < k:
            self.df['similarity_score'] = 0
            return

        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(feature_matrix)), metric='euclidean')
        nbrs.fit(feature_matrix)

        distances, indices = nbrs.kneighbors(feature_matrix)

        similarity_scores = []
        for i, dist_row in enumerate(distances):
            avg_distance = np.mean(dist_row[1:])
            similarity_score = 1 / (1 + avg_distance)
            similarity_scores.append(similarity_score)

        self.df['similarity_score'] = similarity_scores

    def add_short_term_momentum(self, windows=[5, 10, 20]):
        """
        短期频率动量特征：最近N期的号码出现频率
        捕捉"热号"和"冷号"的短期趋势
        """
        for window in windows:
            front_momentum_col = f'front_momentum_{window}'
            back_momentum_col = f'back_momentum_{window}'

            front_momentum = []
            back_momentum = []

            for idx in range(len(self.df)):
                start_idx = max(0, idx - window)
                recent_df = self.df.iloc[start_idx:idx]

                front_nums = []
                back_nums = []
                if self.lottery_type == 'kl8':
                    for _, row in recent_df.iterrows():
                        nums = row.get('num_list', [])
                        if isinstance(nums, str):
                            nums = [int(x) for x in nums.split(',') if x.strip()]
                        front_nums.extend(nums)
                else:
                    for _, row in recent_df.iterrows():
                        front_nums.extend([row[col] for col in self.front_cols])
                        if self.back_cols:
                            back_nums.extend([row[col] for col in self.back_cols])

                current_front = []
                current_back = []
                if self.lottery_type != 'kl8' and idx < len(self.df):
                    current_front = [self.df.iloc[idx][col] for col in self.front_cols]
                    if self.back_cols:
                        current_back = [self.df.iloc[idx][col] for col in self.back_cols]

                front_hit_count = sum(1 for num in current_front if num in front_nums)
                front_momentum.append(front_hit_count / (len(self.front_cols) if self.front_cols else 20))
                if self.back_cols:
                    back_hit_count = sum(1 for num in current_back if num in back_nums)
                    back_momentum.append(back_hit_count / len(self.back_cols))

            self.df[front_momentum_col] = front_momentum
            if self.back_cols:
                self.df[back_momentum_col] = back_momentum

            # 防数据泄漏：上述计算用到了本期(idx)开奖号码（预测目标），
            # 整体下移一期，使第 idx 行只携带 idx 之前已知的信息。
            self.df[front_momentum_col] = self.df[front_momentum_col].shift(1).fillna(0)
            if self.back_cols:
                self.df[back_momentum_col] = self.df[back_momentum_col].shift(1).fillna(0)

    def add_conditional_gap_features(self):
        """
        条件间隔特征：特定条件下号码的出现间隔
        例如：奇数号码连续不出现的期数、大号连续不出现的期数
        """
        if self.lottery_type == 'kl8':
            return

        odd_gap = 0
        even_gap = 0
        large_gap = 0
        small_gap = 0

        odd_gaps = []
        even_gaps = []
        large_gaps = []
        small_gaps = []

        threshold = (self.front_range[0] + self.front_range[1]) // 2

        for idx, row in self.df.iterrows():
            front_nums = [row[col] for col in self.front_cols]

            has_odd = any(num % 2 == 1 for num in front_nums)
            has_even = any(num % 2 == 0 for num in front_nums)
            has_large = any(num > threshold for num in front_nums)
            has_small = any(num <= threshold for num in front_nums)

            if has_odd:
                odd_gap = 0
            else:
                odd_gap += 1

            if has_even:
                even_gap = 0
            else:
                even_gap += 1

            if has_large:
                large_gap = 0
            else:
                large_gap += 1

            if has_small:
                small_gap = 0
            else:
                small_gap += 1

            odd_gaps.append(odd_gap)
            even_gaps.append(even_gap)
            large_gaps.append(large_gap)
            small_gaps.append(small_gap)

        self.df['odd_gap'] = odd_gaps
        self.df['even_gap'] = even_gaps
        self.df['large_gap'] = large_gaps
        self.df['small_gap'] = small_gaps

    def add_transition_features(self):
        """
        期间转换特征：号码之间的转换概率
        从上期号码预测本期号码的相关性
        """
        if self.lottery_type == 'kl8':
            return

        prev_front_overlap = []
        prev_back_overlap = []
        transition_score = []

        for idx in range(len(self.df)):
            if idx == 0:
                prev_front_overlap.append(0)
                prev_back_overlap.append(0)
                transition_score.append(0)
                continue

            curr_front = set([self.df.iloc[idx][col] for col in self.front_cols])
            curr_back = set([self.df.iloc[idx][col] for col in self.back_cols]) if self.back_cols else set()

            prev_front = set([self.df.iloc[idx-1][col] for col in self.front_cols])
            prev_back = set([self.df.iloc[idx-1][col] for col in self.back_cols]) if self.back_cols else set()

            front_overlap = len(curr_front & prev_front)
            back_overlap = len(curr_back & prev_back) if self.back_cols else 0

            trans_score = (front_overlap / len(self.front_cols) + back_overlap / len(self.back_cols)) / 2 if self.back_cols else front_overlap / len(self.front_cols)

            prev_front_overlap.append(front_overlap)
            prev_back_overlap.append(back_overlap)
            transition_score.append(trans_score)

        self.df['prev_front_overlap'] = prev_front_overlap
        if self.back_cols:
            self.df['prev_back_overlap'] = prev_back_overlap
        self.df['transition_score'] = transition_score

        # 防数据泄漏：overlap/transition 用到了本期(idx)开奖号码，整体下移一期
        self.df['prev_front_overlap'] = self.df['prev_front_overlap'].shift(1).fillna(0)
        if self.back_cols:
            self.df['prev_back_overlap'] = self.df['prev_back_overlap'].shift(1).fillna(0)
        self.df['transition_score'] = self.df['transition_score'].shift(1).fillna(0)

    # ========== 新增高阶特征 ==========

    def add_omission_distribution_features(self):
        """
        遗漏分布高阶特征：偏度、峰度
        """
        if self.lottery_type == 'kl8':
            return

        omission_skew = []
        omission_kurt = []

        for idx, _ in self.df.iterrows():
            # 计算到当前期为止所有号码遗漏值的分布统计
            all_omissions = []
            for num in range(self.front_range[0], self.front_range[1]):
                # 这里简化：使用当前累积遗漏值
                all_omissions.append(idx  # placeholder - actual computation would need cumulative tracking
                            if False else 0)

        # 实际计算需要在add_omission_values中累积
        # 这里添加骨架，后续可以完善
        pass

    def add_tail_features(self):
        """
        尾数特征：统计0-9尾数出现个数、同尾数计数
        """
        tail_counts_0 = []
        tail_counts_1 = []
        tail_counts_2 = []
        tail_counts_3 = []
        tail_counts_4 = []
        tail_counts_5 = []
        tail_counts_6 = []
        tail_counts_7 = []
        tail_counts_8 = []
        tail_counts_9 = []
        max_same_tail = []
        tail_mod3_counts = [[], [], []]

        for _, row in self.df.iterrows():
            if self.lottery_type == 'kl8':
                nums = row.get('num_list', [])
                if isinstance(nums, str):
                    nums = [int(x) for x in nums.split(',') if x.strip()]
            else:
                nums = [row[col] for col in self.front_cols]

            tails = [num % 10 for num in nums]
            cnt = Counter(tails)

            tail_counts_0.append(cnt.get(0, 0))
            tail_counts_1.append(cnt.get(1, 0))
            tail_counts_2.append(cnt.get(2, 0))
            tail_counts_3.append(cnt.get(3, 0))
            tail_counts_4.append(cnt.get(4, 0))
            tail_counts_5.append(cnt.get(5, 0))
            tail_counts_6.append(cnt.get(6, 0))
            tail_counts_7.append(cnt.get(7, 0))
            tail_counts_8.append(cnt.get(8, 0))
            tail_counts_9.append(cnt.get(9, 0))
            max_same_tail.append(max(cnt.values()) if cnt else 0)

            for m in range(3):
                cnt_mod3 = sum(1 for t in tails if t % 3 == m)
                tail_mod3_counts[m].append(cnt_mod3)

        self.df['tail_0'] = tail_counts_0
        self.df['tail_1'] = tail_counts_1
        self.df['tail_2'] = tail_counts_2
        self.df['tail_3'] = tail_counts_3
        self.df['tail_4'] = tail_counts_4
        self.df['tail_5'] = tail_counts_5
        self.df['tail_6'] = tail_counts_6
        self.df['tail_7'] = tail_counts_7
        self.df['tail_8'] = tail_counts_8
        self.df['tail_9'] = tail_counts_9
        self.df['max_same_tail'] = max_same_tail
        for m in range(3):
            self.df[f'tail_mod3_{m}'] = tail_mod3_counts[m]

    def add_sum_trend_features(self, windows=[5, 10]):
        """
        和值趋势特征：最近N期和值的移动平均、偏差
        """
        for window in windows:
            ma_col = f'sum_ma_{window}'
            dev_col = f'sum_dev_{window}'
            ma_values = []
            dev_values = []

            for idx in range(len(self.df)):
                start_idx = max(0, idx - window + 1)
                window_sums = self.df['front_sum'].iloc[start_idx:idx+1]
                if len(window_sums) > 0:
                    ma = window_sums.mean()
                    dev = abs(self.df.iloc[idx]['front_sum'] - ma)
                else:
                    ma = 0
                    dev = 0
                ma_values.append(ma)
                dev_values.append(dev)

            self.df[ma_col] = ma_values
            self.df[dev_col] = dev_values

    def add_ac_span_trend_features(self, window=10):
        """
        AC值和跨度趋势特征
        """
        if self.lottery_type == 'kl8':
            return

        ac_ma = []
        span_ma = []

        for idx in range(len(self.df)):
            start_idx = max(0, idx - window + 1)
            if 'ac_value' in self.df.columns:
                ac_window = self.df['ac_value'].iloc[start_idx:idx+1]
                ac_ma.append(ac_window.mean() if len(ac_window) > 0 else 0)
            if 'span' in self.df.columns:
                span_window = self.df['span'].iloc[start_idx:idx+1]
                span_ma.append(span_window.mean() if len(span_window) > 0 else 0)

        if 'ac_value' in self.df.columns:
            self.df[f'ac_ma_{window}'] = ac_ma
        if 'span' in self.df.columns:
            self.df[f'span_ma_{window}'] = span_ma

    def add_edge_code_features(self):
        """
        边码特征：边码 = 上期开奖号码 ±1
        统计本期开出了几个边码
        """
        if self.lottery_type == 'kl8':
            return

        edge_counts = []

        for idx in range(len(self.df)):
            if idx == 0:
                edge_counts.append(0)
                continue

            # 获取上期号码
            prev_nums = []
            for col in self.front_cols:
                prev_nums.append(self.df.iloc[idx-1][col])

            # 生成边码集合
            edge_codes = set()
            for num in prev_nums:
                if self.front_range[0] < num < self.front_range[1] - 1:
                    edge_codes.add(num - 1)
                    edge_codes.add(num + 1)

            # 统计本期边码出现次数
            curr_nums = [self.df.iloc[idx][col] for col in self.front_cols]
            count = sum(1 for num in curr_nums if num in edge_codes)
            edge_counts.append(count)

        self.df['edge_code_count'] = edge_counts
        # 防数据泄漏：edge_code_count 统计了本期(idx)开奖号码，整体下移一期
        self.df['edge_code_count'] = self.df['edge_code_count'].shift(1).fillna(0)

    def add_repeat_number_features(self):
        """
        重号特征：与上期重复号码个数
        """
        if self.lottery_type == 'kl8':
            # 快乐8需要这个特征，因为快乐8出重号很常见
            repeat_counts = []
            for idx in range(len(self.df)):
                if idx == 0:
                    repeat_counts.append(0)
                    continue

                prev_nums = self.df.iloc[idx-1].get('num_list', [])
                if isinstance(prev_nums, str):
                    prev_nums = [int(x) for x in prev_nums.split(',') if x.strip()]
                prev_set = set(prev_nums)

                curr_nums = self.df.iloc[idx].get('num_list', [])
                if isinstance(curr_nums, str):
                    curr_nums = [int(x) for x in curr_nums.split(',') if x.strip()]
                curr_set = set(curr_nums)

                repeat_counts.append(len(curr_set & prev_set))

            self.df['repeat_from_last'] = repeat_counts
            # 防数据泄漏：重号数用到了本期(idx)开奖号码，整体下移一期
            self.df['repeat_from_last'] = self.df['repeat_from_last'].shift(1).fillna(0)
            return

        repeat_counts = []
        for idx in range(len(self.df)):
            if idx == 0:
                repeat_counts.append(0)
                continue

            prev_nums = set(self.df.iloc[idx-1][col] for col in self.front_cols)
            curr_nums = set(self.df.iloc[idx][col] for col in self.front_cols)
            repeat_counts.append(len(curr_nums & prev_nums))

        self.df['repeat_from_last'] = repeat_counts
        # 防数据泄漏：重号数用到了本期(idx)开奖号码，整体下移一期
        self.df['repeat_from_last'] = self.df['repeat_from_last'].shift(1).fillna(0)


if __name__ == "__main__":
    from database import Database

    db = Database()
    df = db.get_all_records()

    if not df.empty:
        fe = FeatureEngineer(df)
        df_features = fe.extract_all_features()
        print("特征工程完成")
        print(f"特征数量: {len(df_features.columns)}")
        print(f"列名: {list(df_features.columns)}")
        print(df_features.head())
