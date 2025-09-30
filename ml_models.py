import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

class LotteryPredictor:
    def __init__(self):
        self.rf_models_front = []
        self.rf_models_back = []
        self.xgb_models_front = []
        self.xgb_models_back = []
        self.scaler = StandardScaler()
        self.label_encoders_front = []
        self.label_encoders_back = []
        
    def prepare_data(self, df):
        feature_cols = ['odd_count', 'even_count', 'large_count', 'small_count', 
                       'front_sum', 'ac_value', 'span', 'max_front_omission', 
                       'avg_front_omission', 'max_back_omission', 'avg_back_omission']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            print("警告：没有可用的特征列，使用基础特征")
            available_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        X = df[available_cols].fillna(0).values
        X = self.scaler.fit_transform(X)
        
        return X
    
    def train_random_forest(self, df):
        if len(df) < 50:
            raise ValueError(f"训练数据不足：需要至少50条数据，当前只有{len(df)}条")
        
        X = self.prepare_data(df)
        
        print(f"开始训练随机森林模型，数据量: {len(df)}条")
        
        for i in range(1, 6):
            y = df[f'front_{i}'].values
            unique_classes = np.unique(y)
            print(f"  前区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X, y)
            self.rf_models_front.append(model)
        
        for i in range(1, 3):
            y = df[f'back_{i}'].values
            unique_classes = np.unique(y)
            print(f"  后区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X, y)
            self.rf_models_back.append(model)
        
        print("✅ 随机森林模型训练完成")
    
    def train_xgboost(self, df):
        if len(df) < 50:
            raise ValueError(f"训练数据不足：需要至少50条数据，当前只有{len(df)}条")
        
        X = self.prepare_data(df)
        
        print(f"开始训练XGBoost模型，数据量: {len(df)}条")
        
        for i in range(1, 6):
            y = df[f'front_{i}'].values
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
            model.fit(X, y_encoded)
            
            self.xgb_models_front.append(model)
            self.label_encoders_front.append(le)
        
        for i in range(1, 3):
            y = df[f'back_{i}'].values
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
            model.fit(X, y_encoded)
            
            self.xgb_models_back.append(model)
            self.label_encoders_back.append(le)
        
        print("✅ XGBoost模型训练完成")
    
    def validate_models(self, df):
        """验证模型是否与当前数据匹配"""
        if not self.rf_models_front or not self.rf_models_back:
            raise ValueError("模型未加载或未训练")
        
        for i in range(1, 6):
            unique_values = df[f'front_{i}'].unique()
            model = self.rf_models_front[i-1]
            
            for val in unique_values[:5]:
                if val not in model.classes_:
                    raise ValueError(
                        f"模型与数据不匹配！前区号码{i}模型不包含数值{val}。"
                        f"模型类别: {model.classes_[:10]}..., 数据包含: {unique_values[:10]}..."
                        f"\n请删除旧模型重新训练。"
                    )
        
        for i in range(1, 3):
            unique_values = df[f'back_{i}'].unique()
            model = self.rf_models_back[i-1]
            
            for val in unique_values[:5]:
                if val not in model.classes_:
                    raise ValueError(
                        f"模型与数据不匹配！后区号码{i}模型不包含数值{val}。"
                        f"\n请删除旧模型重新训练。"
                    )
        
        print("✅ 模型验证通过（前区+后区）")
        return True
    
    def predict_with_probabilities(self, latest_features, play_type='basic'):
        """
        预测号码并返回概率值
        play_type: 'basic' (基础5+2), 'multiple' (多组推荐), 'top_prob' (按概率推荐)
        """
        if isinstance(latest_features, pd.DataFrame):
            latest_features = latest_features.values
        
        if len(latest_features.shape) == 1:
            latest_features = latest_features.reshape(1, -1)
        
        latest_features = self.scaler.transform(latest_features)
        
        front_proba_dict = {}
        back_proba_dict = {}
        
        for model in self.rf_models_front:
            pred_proba = model.predict_proba(latest_features)[0]
            for cls, prob in zip(model.classes_, pred_proba):
                num = int(cls)
                if num not in front_proba_dict:
                    front_proba_dict[num] = []
                front_proba_dict[num].append(prob)
        
        for idx, model in enumerate(self.xgb_models_front):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_front[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in front_proba_dict:
                    front_proba_dict[num] = []
                front_proba_dict[num].append(prob)
        
        for num in front_proba_dict:
            front_proba_dict[num] = np.mean(front_proba_dict[num])
        
        for model in self.rf_models_back:
            pred_proba = model.predict_proba(latest_features)[0]
            for cls, prob in zip(model.classes_, pred_proba):
                num = int(cls)
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob)
        
        for idx, model in enumerate(self.xgb_models_back):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_back[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob)
        
        for num in back_proba_dict:
            back_proba_dict[num] = np.mean(back_proba_dict[num])
        
        sorted_front = sorted(front_proba_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_back = sorted(back_proba_dict.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'play_type': play_type,
            'predictions': []
        }
        
        if play_type == 'basic':
            result['predictions'].append({
                'front': sorted([num for num, _ in sorted_front[:5]]),
                'back': sorted([num for num, _ in sorted_back[:2]]),
                'front_probs': {num: float(prob) for num, prob in sorted_front[:5]},
                'back_probs': {num: float(prob) for num, prob in sorted_back[:2]}
            })
        
        elif play_type == 'multiple':
            for i in range(3):
                start_idx = i * 2
                front_slice = sorted_front[start_idx:start_idx+5]
                back_slice = sorted_back[:2] if i == 0 else sorted_back[min(i, len(sorted_back)-2):min(i+2, len(sorted_back))]
                
                if len(front_slice) >= 5 and len(back_slice) >= 2:
                    result['predictions'].append({
                        'front': sorted([num for num, _ in front_slice[:5]]),
                        'back': sorted([num for num, _ in back_slice[:2]]),
                        'front_probs': {num: float(prob) for num, prob in front_slice[:5]},
                        'back_probs': {num: float(prob) for num, prob in back_slice[:2]}
                    })
        
        elif play_type == 'top_prob':
            result['predictions'].append({
                'front': sorted([num for num, _ in sorted_front[:5]]),
                'back': sorted([num for num, _ in sorted_back[:2]]),
                'front_probs': {num: float(prob) for num, prob in sorted_front[:10]},
                'back_probs': {num: float(prob) for num, prob in sorted_back[:5]}
            })
        
        return result
    
    def predict_numbers(self, latest_features):
        if isinstance(latest_features, pd.DataFrame):
            latest_features = latest_features.values
        
        if len(latest_features.shape) == 1:
            latest_features = latest_features.reshape(1, -1)
        
        latest_features = self.scaler.transform(latest_features)
        
        front_predictions = []
        back_predictions = []
        
        for model in self.rf_models_front:
            pred_proba = model.predict_proba(latest_features)[0]
            top_indices = np.argsort(pred_proba)[-3:][::-1]
            top_nums = model.classes_[top_indices]
            front_predictions.extend(top_nums)
        
        for idx, model in enumerate(self.xgb_models_front):
            pred_proba = model.predict_proba(latest_features)[0]
            top_indices = np.argsort(pred_proba)[-3:][::-1]
            
            le = self.label_encoders_front[idx]
            top_nums = le.inverse_transform(top_indices)
            front_predictions.extend(top_nums)
        
        front_counter = {}
        for num in front_predictions:
            if num not in front_counter:
                front_counter[num] = 0
            front_counter[num] += 1
        
        sorted_front = sorted(front_counter.items(), key=lambda x: x[1], reverse=True)
        final_front = sorted([int(num) for num, _ in sorted_front[:5]])
        
        for model in self.rf_models_back:
            pred_proba = model.predict_proba(latest_features)[0]
            top_indices = np.argsort(pred_proba)[-2:][::-1]
            top_nums = model.classes_[top_indices]
            back_predictions.extend(top_nums)
        
        for idx, model in enumerate(self.xgb_models_back):
            pred_proba = model.predict_proba(latest_features)[0]
            top_indices = np.argsort(pred_proba)[-2:][::-1]
            
            le = self.label_encoders_back[idx]
            top_nums = le.inverse_transform(top_indices)
            back_predictions.extend(top_nums)
        
        back_counter = {}
        for num in back_predictions:
            if num not in back_counter:
                back_counter[num] = 0
            back_counter[num] += 1
        
        sorted_back = sorted(back_counter.items(), key=lambda x: x[1], reverse=True)
        final_back = sorted([int(num) for num, _ in sorted_back[:2]])
        
        return {
            'front': final_front,
            'back': final_back
        }
    
    def save_models(self, path='models'):
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}/rf_front.pkl', 'wb') as f:
            pickle.dump(self.rf_models_front, f)
        with open(f'{path}/rf_back.pkl', 'wb') as f:
            pickle.dump(self.rf_models_back, f)
        with open(f'{path}/xgb_front.pkl', 'wb') as f:
            pickle.dump(self.xgb_models_front, f)
        with open(f'{path}/xgb_back.pkl', 'wb') as f:
            pickle.dump(self.xgb_models_back, f)
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{path}/label_encoders_front.pkl', 'wb') as f:
            pickle.dump(self.label_encoders_front, f)
        with open(f'{path}/label_encoders_back.pkl', 'wb') as f:
            pickle.dump(self.label_encoders_back, f)
        
        print(f"✅ 模型已保存到 {path}")
    
    def load_models(self, path='models'):
        with open(f'{path}/rf_front.pkl', 'rb') as f:
            self.rf_models_front = pickle.load(f)
        with open(f'{path}/rf_back.pkl', 'rb') as f:
            self.rf_models_back = pickle.load(f)
        with open(f'{path}/xgb_front.pkl', 'rb') as f:
            self.xgb_models_front = pickle.load(f)
        with open(f'{path}/xgb_back.pkl', 'rb') as f:
            self.xgb_models_back = pickle.load(f)
        with open(f'{path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{path}/label_encoders_front.pkl', 'rb') as f:
            self.label_encoders_front = pickle.load(f)
        with open(f'{path}/label_encoders_back.pkl', 'rb') as f:
            self.label_encoders_back = pickle.load(f)
        
        print(f"✅ 模型已从 {path} 加载")

if __name__ == "__main__":
    from database import Database
    from feature_engineering import FeatureEngineer
    
    db = Database()
    df = db.get_all_records()
    
    if not df.empty:
        fe = FeatureEngineer(df)
        df_features = fe.extract_all_features()
        
        predictor = LotteryPredictor()
        predictor.train_random_forest(df_features)
        predictor.train_xgboost(df_features)
        predictor.save_models()
        
        print("模型训练和保存完成")
