import pandas as pd
import numpy as np
from collections import Counter

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
