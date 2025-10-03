import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
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
        self.add_consecutive_features()
        self.add_interval_density()
        self.add_volatility_features()
        self.add_similarity_features()
        
        return self.df
    
    def add_number_frequency(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        back_cols = ['back_1', 'back_2']
        
        front_freq = {}
        back_freq = {}
        
        for num in range(1, 36):
            front_freq[num] = 0
        for num in range(1, 13):
            back_freq[num] = 0
        
        for _, row in self.df.iterrows():
            for col in front_cols:
                front_freq[row[col]] += 1
            for col in back_cols:
                back_freq[row[col]] += 1
        
        self.front_freq = front_freq
        self.back_freq = back_freq
    
    def add_omission_values(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        front_omission = {i: 0 for i in range(1, 36)}
        back_omission = {i: 0 for i in range(1, 13)}
        
        omission_records = []
        
        for idx, row in self.df.iterrows():
            front_nums = [row[col] for col in front_cols]
            back_nums = [row['back_1'], row['back_2']]
            
            for num in range(1, 36):
                if num in front_nums:
                    front_omission[num] = 0
                else:
                    front_omission[num] += 1
            
            for num in range(1, 13):
                if num in back_nums:
                    back_omission[num] = 0
                else:
                    back_omission[num] += 1
            
            omission_records.append({
                'max_front_omission': max(front_omission.values()),
                'avg_front_omission': np.mean(list(front_omission.values())),
                'max_back_omission': max(back_omission.values()),
                'avg_back_omission': np.mean(list(back_omission.values()))
            })
        
        for key in omission_records[0].keys():
            self.df[key] = [r[key] for r in omission_records]
    
    def add_odd_even_ratio(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        odd_counts = []
        even_counts = []
        
        for _, row in self.df.iterrows():
            odd = sum(1 for col in front_cols if row[col] % 2 == 1)
            even = 5 - odd
            odd_counts.append(odd)
            even_counts.append(even)
        
        self.df['odd_count'] = odd_counts
        self.df['even_count'] = even_counts
    
    def add_size_ratio(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        large_counts = []
        small_counts = []
        
        for _, row in self.df.iterrows():
            large = sum(1 for col in front_cols if row[col] > 18)
            small = 5 - large
            large_counts.append(large)
            small_counts.append(small)
        
        self.df['large_count'] = large_counts
        self.df['small_count'] = small_counts
    
    def add_sum_values(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        sums = []
        for _, row in self.df.iterrows():
            total = sum(row[col] for col in front_cols)
            sums.append(total)
        
        self.df['front_sum'] = sums
    
    def add_ac_value(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        ac_values = []
        for _, row in self.df.iterrows():
            nums = sorted([row[col] for col in front_cols])
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(abs(nums[i] - nums[j]))
            ac = len(diffs) - 4
            ac_values.append(ac)
        
        self.df['ac_value'] = ac_values
    
    def add_span_value(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        spans = []
        for _, row in self.df.iterrows():
            nums = [row[col] for col in front_cols]
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
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        pair_counts = Counter()
        triplet_counts = Counter()
        total_draws = len(self.df)
        
        for _, row in self.df.iterrows():
            nums = sorted([row[col] for col in front_cols])
            
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
            nums = sorted([row[col] for col in front_cols])
            
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
            
            p_xy = max_pair_freq / total_draws if total_draws > 0 else 0
            p_x = self.front_freq.get(nums[0], 0) / (total_draws * 5) if total_draws > 0 else 0
            p_y = self.front_freq.get(nums[1], 0) / (total_draws * 5) if total_draws > 0 else 0
            
            pmi = np.log2(p_xy / (p_x * p_y + 1e-10) + 1e-10) if p_xy > 0 else 0
            
            pair_scores.append(max_pair_freq)
            triplet_scores.append(max_triplet_freq)
            pmi_scores.append(pmi)
        
        self.df['max_pair_freq'] = pair_scores
        self.df['max_triplet_freq'] = triplet_scores
        self.df['pmi_score'] = pmi_scores
    
    def add_temporal_features(self):
        if 'draw_date' not in self.df.columns:
            return
        
        weekdays = []
        months = []
        weekday_sin = []
        weekday_cos = []
        month_sin = []
        month_cos = []
        
        for _, row in self.df.iterrows():
            try:
                if pd.isna(row['draw_date']):
                    weekday = 0
                    month = 1
                else:
                    date_obj = pd.to_datetime(row['draw_date'])
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
    
    def add_consecutive_features(self):
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        consecutive_counts = []
        max_consecutive = []
        
        for _, row in self.df.iterrows():
            nums = sorted([row[col] for col in front_cols])
            
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
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        zone1_counts = []
        zone2_counts = []
        zone3_counts = []
        
        for _, row in self.df.iterrows():
            nums = [row[col] for col in front_cols]
            
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
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        rolling_std = []
        rolling_entropy = []
        
        for idx in range(len(self.df)):
            start_idx = max(0, idx - window + 1)
            window_data = self.df.iloc[start_idx:idx+1]
            
            all_nums = []
            for _, row in window_data.iterrows():
                all_nums.extend([row[col] for col in front_cols])
            
            if len(all_nums) > 0:
                std_val = np.std(all_nums)
                
                num_counts = np.bincount(all_nums, minlength=36)[1:36]
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
        front_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        feature_matrix = self.df[front_cols].values
        
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

if __name__ == "__main__":
    from database import Database
    
    db = Database()
    df = db.get_all_records()
    
    if not df.empty:
        fe = FeatureEngineer(df)
        df_features = fe.extract_all_features()
        print("特征工程完成")
        print(df_features.columns.tolist())
        print(df_features.head())
