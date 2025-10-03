import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
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
        
        self.rf_weight = 0.2
        self.xgb_weight = 0.2
        self.lstm_weight = 0.3
        self.transformer_weight = 0.3
        
        self.dl_predictor = None
        self.use_deep_learning = False
        
    def prepare_data(self, df):
        base_features = ['odd_count', 'even_count', 'large_count', 'small_count', 
                        'front_sum', 'ac_value', 'span', 'max_front_omission', 
                        'avg_front_omission', 'max_back_omission', 'avg_back_omission']
        
        advanced_features = ['max_pair_freq', 'max_triplet_freq', 'pmi_score',
                           'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
                           'consecutive_count', 'max_consecutive',
                           'zone1_count', 'zone2_count', 'zone3_count',
                           'rolling_std', 'rolling_entropy', 'similarity_score']
        
        feature_cols = base_features + advanced_features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            print("警告：没有可用的特征列，使用基础特征")
            available_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        print(f"使用 {len(available_cols)} 个特征: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")
        
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
    
    def train_stacking(self, df):
        """配置集成策略（深度学习模型需单独训练）"""
        print(f"开始配置集成策略，数据量: {len(df)}条")
        
        self.full_df = df.copy()
        
        print(f"⚠️  深度学习模型训练耗时较长（10-20分钟），请运行: python deep_learning_models.py")
        print(f"✅ 集成策略配置完成（RF={self.rf_weight}, XGB={self.xgb_weight}）")
        print(f"   如需启用深度学习，训练完成后会自动加载LSTM和Transformer模型")
    
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
                front_proba_dict[num].append(prob * self.rf_weight)
        
        for idx, model in enumerate(self.xgb_models_front):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_front[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in front_proba_dict:
                    front_proba_dict[num] = []
                front_proba_dict[num].append(prob * self.xgb_weight)
        
        for model in self.rf_models_back:
            pred_proba = model.predict_proba(latest_features)[0]
            for cls, prob in zip(model.classes_, pred_proba):
                num = int(cls)
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob * self.rf_weight)
        
        for idx, model in enumerate(self.xgb_models_back):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_back[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob * self.xgb_weight)
        
        if self.use_deep_learning and self.dl_predictor:
            try:
                dl_probs = self.dl_predictor.predict_with_probabilities(self.full_df)
                
                for num, prob in dl_probs['front_lstm'].items():
                    if num not in front_proba_dict:
                        front_proba_dict[num] = []
                    front_proba_dict[num].append(prob * self.lstm_weight)
                
                for num, prob in dl_probs['front_transformer'].items():
                    if num not in front_proba_dict:
                        front_proba_dict[num] = []
                    front_proba_dict[num].append(prob * self.transformer_weight)
                
                for num, prob in dl_probs['back_lstm'].items():
                    if num not in back_proba_dict:
                        back_proba_dict[num] = []
                    back_proba_dict[num].append(prob * self.lstm_weight)
                
                for num, prob in dl_probs['back_transformer'].items():
                    if num not in back_proba_dict:
                        back_proba_dict[num] = []
                    back_proba_dict[num].append(prob * self.transformer_weight)
            except Exception as e:
                print(f"⚠️  深度学习预测失败: {e}")
        
        for num in front_proba_dict:
            front_proba_dict[num] = np.mean(front_proba_dict[num])
        
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
                'front': sorted([num for num, _ in sorted_front[:10]]),
                'back': sorted([num for num, _ in sorted_back[:5]]),
                'front_probs': {num: float(prob) for num, prob in sorted_front[:10]},
                'back_probs': {num: float(prob) for num, prob in sorted_back[:5]}
            })
        
        elif play_type == 'dantuo':
            front_dan = sorted([num for num, _ in sorted_front[:2]])
            front_tuo = sorted([num for num, _ in sorted_front[2:10]])
            
            back_dan = sorted([num for num, _ in sorted_back[:1]])
            back_tuo = sorted([num for num, _ in sorted_back[1:5]])
            
            result['predictions'].append({
                'front_dan': front_dan,
                'front_tuo': front_tuo,
                'back_dan': back_dan,
                'back_tuo': back_tuo,
                'front_dan_probs': {num: float(prob) for num, prob in sorted_front[:2]},
                'front_tuo_probs': {num: float(prob) for num, prob in sorted_front[2:10]},
                'back_dan_probs': {num: float(prob) for num, prob in sorted_back[:1]},
                'back_tuo_probs': {num: float(prob) for num, prob in sorted_back[1:5]},
                'combination_count': self._calculate_dantuo_combinations(len(front_dan), len(front_tuo), len(back_dan), len(back_tuo)),
                'description': f'前区胆{len(front_dan)}拖{len(front_tuo)}，后区胆{len(back_dan)}拖{len(back_tuo)}'
            })
        
        return result
    
    def _calculate_dantuo_combinations(self, front_dan, front_tuo, back_dan, back_tuo):
        """计算胆拖组合数量"""
        from math import comb
        front_need = 5 - front_dan
        back_need = 2 - back_dan
        
        if front_tuo < front_need or back_tuo < back_need:
            return 0
        
        front_combos = comb(front_tuo, front_need)
        back_combos = comb(back_tuo, back_need)
        
        return front_combos * back_combos
    
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
        
        weights_dict = {
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight,
            'transformer_weight': self.transformer_weight
        }
        with open(f'{path}/ensemble_weights.pkl', 'wb') as f:
            pickle.dump(weights_dict, f)
        
        model_info = f"RF={self.rf_weight}, XGB={self.xgb_weight}"
        if self.use_deep_learning:
            model_info += f", LSTM={self.lstm_weight}, Transformer={self.transformer_weight}"
        print(f"✅ 模型已保存到 {path} ({model_info})")
    
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
        
        if os.path.exists(f'{path}/ensemble_weights.pkl'):
            with open(f'{path}/ensemble_weights.pkl', 'rb') as f:
                weights = pickle.load(f)
                self.rf_weight = weights.get('rf_weight', 0.2)
                self.xgb_weight = weights.get('xgb_weight', 0.2)
                self.lstm_weight = weights.get('lstm_weight', 0.3)
                self.transformer_weight = weights.get('transformer_weight', 0.3)
            print(f"✅ 模型已从 {path} 加载 (RF={self.rf_weight}, XGB={self.xgb_weight}, LSTM={self.lstm_weight}, Transformer={self.transformer_weight})")
        else:
            print(f"✅ 模型已从 {path} 加载 (使用默认权重)")
        
        if os.path.exists(f'{path}/dl_config.pkl'):
            try:
                from deep_learning_models import DeepLearningPredictor
                self.dl_predictor = DeepLearningPredictor()
                if self.dl_predictor.load_models(path):
                    self.use_deep_learning = True
                else:
                    self.use_deep_learning = False
            except Exception as e:
                print(f"⚠️  深度学习模型加载失败: {e}")
                self.use_deep_learning = False
        else:
            self.use_deep_learning = False

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
