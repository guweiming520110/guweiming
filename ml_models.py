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
from feature_config import get_all_features

class LotteryPredictor:
    def __init__(self, lottery_type='dlt'):
        """
        初始化彩票预测器
        
        参数:
            lottery_type: 彩票类型 (dlt=大乐透, ssq=双色球, fc3d=福彩3D)
        """
        self.lottery_type = lottery_type
        self.rf_models_front = []
        self.rf_models_back = []
        self.xgb_models_front = []
        self.xgb_models_back = []
        self.scaler = StandardScaler()
        self.label_encoders_front = []
        self.label_encoders_back = []
        # 训练时实际使用的特征列顺序，预测时必须严格对齐，防止特征漂移导致 scaler 错位
        self.feature_cols = None
        
        self.rf_weight = 0.4
        self.xgb_weight = 0.6
        self.lstm_weight = 0.0
        self.transformer_weight = 0.0
        
        self.dl_predictor = None
        self.use_deep_learning = False
        # 深度学习预测所需的完整历史数据，预测前需由调用方注入（见 set_full_df）
        self.full_df = None
        
        # 根据彩票类型设置号码数量
        if lottery_type == 'dlt':
            self.front_count = 5
            self.back_count = 2
        elif lottery_type == 'ssq':
            self.front_count = 6
            self.back_count = 1
        elif lottery_type == 'fc3d':
            raise NotImplementedError("福彩3D 尚未完整实现，暂不支持")
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")
        
    def prepare_data(self, df, fit=True):
        feature_cols = get_all_features()
        available_cols = [col for col in feature_cols if col in df.columns]

        if not available_cols:
            print("警告：没有可用的特征列，使用基础特征")
            # 动态生成基础特征列名，基于当前彩票类型
            available_cols = [f'front_{i}' for i in range(1, self.front_count + 1)]
            if self.back_count > 0:
                available_cols += [f'back_{i}' for i in range(1, self.back_count + 1)]

        # 记录训练时的特征列顺序，供预测/持久化对齐使用
        if fit:
            self.feature_cols = available_cols

        print(f"使用 {len(available_cols)} 个特征: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")

        X = df[available_cols].fillna(0).values
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X
    
    def _report_validation_accuracy(self, tag, y_pred, y_true):
        """打印单个号码位置的验证集 Top-1 准确率，返回准确率(0-1)或 None。

        y_pred/y_true 长度理论上一致（同一验证段构造），防御性截断到较短者，
        避免标签列含 NaN 被 dropna 后长度不匹配导致广播报错。
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        n = min(len(y_pred), len(y_true))
        if n == 0:
            print(f"  验证集{tag}: 无验证样本，跳过评估")
            return None
        y_pred, y_true = y_pred[:n], y_true[:n]
        hit = int(np.sum(y_pred == y_true))
        acc = hit / n
        print(f"  验证集{tag} Top-1准确率: {acc * 100:.1f}% ({hit}/{n})")
        return acc

    def train_random_forest(self, df):
        if len(df) < 50:
            raise ValueError(f"训练数据不足：需要至少50条数据，当前只有{len(df)}条")

        df = df.copy().reset_index(drop=True)

        # H6 修复: 按时间序划分训练/验证集——最后20%(至少1条)作验证，前80%训练。
        # df 已按 period 升序，验证段行严格晚于训练段行，特征/标签均不跨段混用。
        n_val = max(1, int(len(df) * 0.2))
        train_df = df.iloc[:-n_val]
        val_df = df.iloc[-n_val:]

        # C1 修复: 特征语义 = 截止当期(info≤t, 含当期开奖)；标签 = 下一期开奖(t+1)。
        # 因此各段剔除最后一行特征(它没有"下一期"标签)。
        feat_train_df = train_df.iloc[:-1]
        X_train = self.prepare_data(feat_train_df)   # scaler 只 fit 训练段，防止验证信息泄漏

        # 验证段复用同一 scaler，仅 transform（self.feature_cols 已由训练段 fit 时设置）
        feat_val_df = val_df.iloc[:-1]
        X_val = self.prepare_data(feat_val_df, fit=False)

        print(f"开始训练随机森林模型，训练样本: {len(feat_train_df)}条, 验证样本: {len(feat_val_df)}条 "
              f"(原{len(df)}条, 时间序前80%/后20%, 各段末行保留作推理最新特征)")

        # 训练前区模型
        front_accs = []
        for i in range(1, self.front_count + 1):
            y = train_df[f'front_{i}'].shift(-1).iloc[:-1].values
            y_val = val_df[f'front_{i}'].shift(-1).dropna().values
            unique_classes = np.unique(y)
            print(f"  前区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")

            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y)
            self.rf_models_front.append(model)
            acc = self._report_validation_accuracy(f"前区{i}号", model.predict(X_val), y_val)
            if acc is not None:
                front_accs.append(acc)

        # 训练后区模型
        back_accs = []
        for i in range(1, self.back_count + 1):
            y = train_df[f'back_{i}'].shift(-1).iloc[:-1].values
            y_val = val_df[f'back_{i}'].shift(-1).dropna().values
            unique_classes = np.unique(y)
            print(f"  后区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")

            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y)
            self.rf_models_back.append(model)
            acc = self._report_validation_accuracy(f"后区{i}号", model.predict(X_val), y_val)
            if acc is not None:
                back_accs.append(acc)

        if front_accs:
            print(f"  验证集前区平均 Top-1准确率: {np.mean(front_accs) * 100:.1f}%")
        if back_accs:
            print(f"  验证集后区平均 Top-1准确率: {np.mean(back_accs) * 100:.1f}%")

        print("✅ 随机森林模型训练完成")

    def train_xgboost(self, df):
        if len(df) < 50:
            raise ValueError(f"训练数据不足：需要至少50条数据，当前只有{len(df)}条")

        df = df.copy().reset_index(drop=True)

        # H6 修复: 同 train_random_forest —— 时间序前80%训练/后20%验证，scaler 只 fit 训练段
        n_val = max(1, int(len(df) * 0.2))
        train_df = df.iloc[:-n_val]
        val_df = df.iloc[-n_val:]

        # C1 修复: 同 train_random_forest —— 特征 info≤t，标签下一期开奖 t+1
        feat_train_df = train_df.iloc[:-1]
        X_train = self.prepare_data(feat_train_df)   # scaler 只 fit 训练段

        # 验证段复用同一 scaler，仅 transform
        feat_val_df = val_df.iloc[:-1]
        X_val = self.prepare_data(feat_val_df, fit=False)

        print(f"开始训练XGBoost模型，训练样本: {len(feat_train_df)}条, 验证样本: {len(feat_val_df)}条 "
              f"(原{len(df)}条, 时间序前80%/后20%, 各段末行保留作推理最新特征)")

        # 训练前区模型
        front_accs = []
        for i in range(1, self.front_count + 1):
            y = train_df[f'front_{i}'].shift(-1).iloc[:-1].values
            y_val = val_df[f'front_{i}'].shift(-1).dropna().values
            unique_classes = np.unique(y)
            print(f"  前区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
            model.fit(X_train, y_encoded)

            self.xgb_models_front.append(model)
            self.label_encoders_front.append(le)

            # XGB 输出编码值，需 inverse_transform 还原真实号码后计算验证准确率
            y_pred = le.inverse_transform(model.predict(X_val))
            acc = self._report_validation_accuracy(f"前区{i}号", y_pred, y_val)
            if acc is not None:
                front_accs.append(acc)

        # 训练后区模型
        back_accs = []
        for i in range(1, self.back_count + 1):
            y = train_df[f'back_{i}'].shift(-1).iloc[:-1].values
            y_val = val_df[f'back_{i}'].shift(-1).dropna().values
            unique_classes = np.unique(y)
            print(f"  后区号码{i}: {len(unique_classes)}个不同值 (范围: {min(unique_classes)}-{max(unique_classes)})")

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
            model.fit(X_train, y_encoded)

            self.xgb_models_back.append(model)
            self.label_encoders_back.append(le)

            y_pred = le.inverse_transform(model.predict(X_val))
            acc = self._report_validation_accuracy(f"后区{i}号", y_pred, y_val)
            if acc is not None:
                back_accs.append(acc)

        if front_accs:
            print(f"  验证集前区平均 Top-1准确率: {np.mean(front_accs) * 100:.1f}%")
        if back_accs:
            print(f"  验证集后区平均 Top-1准确率: {np.mean(back_accs) * 100:.1f}%")

        print("✅ XGBoost模型训练完成")
    
    def train_stacking(self, df):
        """配置集成策略（深度学习模型需单独训练）"""
        print(f"开始配置集成策略，数据量: {len(df)}条")
        
        self.full_df = df.copy()
        
        print(f"⚠️  深度学习模型训练耗时较长（10-20分钟），请运行: python deep_learning_models.py")
        print(f"✅ 集成策略配置完成（RF={self.rf_weight}, XGB={self.xgb_weight}）")
        print(f"   如需启用深度学习，训练完成后会自动加载LSTM和Transformer模型")

    def set_full_df(self, df):
        """注入深度学习预测所需的完整历史数据。

        load_models 不会恢复 full_df（仅 train_stacking 会设置），
        因此从磁盘加载模型后启用深度学习时，必须由调用方显式注入，
        否则 predict_with_probabilities 会因 full_df 为 None 跳过 LSTM/Transformer。
        """
        self.full_df = df.copy() if df is not None else None

    def validate_models(self, df):
        """验证模型是否与当前数据匹配"""
        if not self.rf_models_front:
            raise ValueError("前区模型未加载或未训练")
        if self.back_count > 0 and not self.rf_models_back:
            raise ValueError("后区模型未加载或未训练")
        
        # 验证前区模型
        for i in range(1, self.front_count + 1):
            unique_values = df[f'front_{i}'].unique()
            model = self.rf_models_front[i-1]
            
            for val in unique_values:
                if val not in model.classes_:
                    raise ValueError(
                        f"模型与数据不匹配！前区号码{i}模型不包含数值{val}。"
                        f"模型类别范围: {model.classes_.min()}-{model.classes_.max()}, 数据包含: {unique_values.min()}-{unique_values.max()}"
                        f"\n请删除旧模型重新训练。"
                    )
        
        # 验证后区模型
        for i in range(1, self.back_count + 1):
            unique_values = df[f'back_{i}'].unique()
            model = self.rf_models_back[i-1]
            
            for val in unique_values:
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
            # 严格对齐训练时的特征列顺序，缺失列补0，避免特征漂移导致 scaler 错位
            if self.feature_cols is not None:
                latest_features = latest_features.reindex(
                    columns=self.feature_cols, fill_value=0
                ).fillna(0)
            latest_features = latest_features.values

        if len(latest_features.shape) == 1:
            latest_features = latest_features.reshape(1, -1)

        latest_features = self.scaler.transform(latest_features)

        # 判定本次预测深度学习是否真正可用：启用且已加载模型且已注入历史数据
        dl_active = bool(self.use_deep_learning and self.dl_predictor and self.full_df is not None)

        # 有效权重：深度学习不可用时，把本应给 LSTM/Transformer 的权重按 RF/XGB 原比例
        # 回收并归一化到 RF/XGB，避免 DL 静默失效时 RF/XGB 权重被无谓稀释（各占0.2导致总和仅0.4）
        rf_weight = self.rf_weight
        xgb_weight = self.xgb_weight
        if not dl_active and (self.lstm_weight or self.transformer_weight):
            if self.use_deep_learning and self.full_df is None:
                print("⚠️  深度学习已启用但未注入历史数据(full_df)，将回收DL权重至RF/XGB")
            ml_total = self.rf_weight + self.xgb_weight
            if ml_total > 0:
                rf_weight = self.rf_weight / ml_total
                xgb_weight = self.xgb_weight / ml_total
            else:
                # RF/XGB 权重均为0的异常配置，退化为均分
                rf_weight = xgb_weight = 0.5

        front_proba_dict = {}
        back_proba_dict = {}

        for model in self.rf_models_front:
            pred_proba = model.predict_proba(latest_features)[0]
            for cls, prob in zip(model.classes_, pred_proba):
                num = int(cls)
                if num not in front_proba_dict:
                    front_proba_dict[num] = []
                front_proba_dict[num].append(prob * rf_weight)

        for idx, model in enumerate(self.xgb_models_front):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_front[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in front_proba_dict:
                    front_proba_dict[num] = []
                front_proba_dict[num].append(prob * xgb_weight)

        for model in self.rf_models_back:
            pred_proba = model.predict_proba(latest_features)[0]
            for cls, prob in zip(model.classes_, pred_proba):
                num = int(cls)
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob * rf_weight)

        for idx, model in enumerate(self.xgb_models_back):
            pred_proba = model.predict_proba(latest_features)[0]
            le = self.label_encoders_back[idx]
            for cls_idx, prob in enumerate(pred_proba):
                num = int(le.inverse_transform([cls_idx])[0])
                if num not in back_proba_dict:
                    back_proba_dict[num] = []
                back_proba_dict[num].append(prob * xgb_weight)

        if dl_active:
            assert self.dl_predictor is not None
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
                'front': sorted([num for num, _ in sorted_front[:self.front_count]]),
                'back': sorted([num for num, _ in sorted_back[:self.back_count]]),
                'front_probs': {num: float(prob) for num, prob in sorted_front[:self.front_count]},
                'back_probs': {num: float(prob) for num, prob in sorted_back[:self.back_count]}
            })
        
        elif play_type == 'multiple':
            for i in range(3):
                start_idx = i * 2
                front_slice = sorted_front[start_idx:start_idx+self.front_count]
                back_slice = sorted_back[:self.back_count] if i == 0 else sorted_back[min(i, len(sorted_back)-self.back_count):min(i+self.back_count, len(sorted_back))]
                
                if len(front_slice) >= self.front_count and len(back_slice) >= self.back_count:
                    result['predictions'].append({
                        'front': sorted([num for num, _ in front_slice[:self.front_count]]),
                        'back': sorted([num for num, _ in back_slice[:self.back_count]]),
                        'front_probs': {num: float(prob) for num, prob in front_slice[:self.front_count]},
                        'back_probs': {num: float(prob) for num, prob in back_slice[:self.back_count]}
                    })
        
        elif play_type == 'top_prob':
            top_front = self.front_count * 2
            top_back = max(self.back_count * 2, 5)
            result['predictions'].append({
                'front': sorted([num for num, _ in sorted_front[:top_front]]),
                'back': sorted([num for num, _ in sorted_back[:top_back]]),
                'front_probs': {num: float(prob) for num, prob in sorted_front[:top_front]},
                'back_probs': {num: float(prob) for num, prob in sorted_back[:top_back]}
            })
        
        elif play_type == 'dantuo':
            front_dan = sorted([num for num, _ in sorted_front[:2]])
            front_tuo = sorted([num for num, _ in sorted_front[2:self.front_count+3]])
            
            back_dan = sorted([num for num, _ in sorted_back[:1]])
            back_tuo = sorted([num for num, _ in sorted_back[1:self.back_count+2]])
            
            result['predictions'].append({
                'front_dan': front_dan,
                'front_tuo': front_tuo,
                'back_dan': back_dan,
                'back_tuo': back_tuo,
                'front_dan_probs': {num: float(prob) for num, prob in sorted_front[:2]},
                'front_tuo_probs': {num: float(prob) for num, prob in sorted_front[2:self.front_count+3]},
                'back_dan_probs': {num: float(prob) for num, prob in sorted_back[:1]},
                'back_tuo_probs': {num: float(prob) for num, prob in sorted_back[1:self.back_count+2]},
                'combination_count': self._calculate_dantuo_combinations(len(front_dan), len(front_tuo), len(back_dan), len(back_tuo)),
                'description': f'前区胆{len(front_dan)}拖{len(front_tuo)}，后区胆{len(back_dan)}拖{len(back_tuo)}'
            })
        
        return result
    
    def _calculate_dantuo_combinations(self, front_dan, front_tuo, back_dan, back_tuo):
        """计算胆拖组合数量"""
        from math import comb
        front_need = self.front_count - front_dan
        back_need = self.back_count - back_dan
        
        if front_tuo < front_need or back_tuo < back_need:
            return 0
        
        front_combos = comb(front_tuo, front_need)
        back_combos = comb(back_tuo, back_need)
        
        return front_combos * back_combos
    
    def predict_numbers(self, latest_features):
        if isinstance(latest_features, pd.DataFrame):
            # 严格对齐训练时的特征列顺序，缺失列补0，避免特征漂移导致 scaler 错位
            if self.feature_cols is not None:
                latest_features = latest_features.reindex(
                    columns=self.feature_cols, fill_value=0
                ).fillna(0)
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
        # 为不同彩票类型使用不同的子目录
        save_path = os.path.join(path, self.lottery_type)
        os.makedirs(save_path, exist_ok=True)
        
        with open(f'{save_path}/rf_front.pkl', 'wb') as f:
            pickle.dump(self.rf_models_front, f)
        with open(f'{save_path}/rf_back.pkl', 'wb') as f:
            pickle.dump(self.rf_models_back, f)
        with open(f'{save_path}/xgb_front.pkl', 'wb') as f:
            pickle.dump(self.xgb_models_front, f)
        with open(f'{save_path}/xgb_back.pkl', 'wb') as f:
            pickle.dump(self.xgb_models_back, f)
        with open(f'{save_path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{save_path}/label_encoders_front.pkl', 'wb') as f:
            pickle.dump(self.label_encoders_front, f)
        with open(f'{save_path}/label_encoders_back.pkl', 'wb') as f:
            pickle.dump(self.label_encoders_back, f)
        with open(f'{save_path}/feature_cols.pkl', 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        weights_dict = {
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight,
            'transformer_weight': self.transformer_weight
        }
        with open(f'{save_path}/ensemble_weights.pkl', 'wb') as f:
            pickle.dump(weights_dict, f)
        
        model_info = f"RF={self.rf_weight}, XGB={self.xgb_weight}"
        if self.use_deep_learning:
            model_info += f", LSTM={self.lstm_weight}, Transformer={self.transformer_weight}"
        print(f"✅ 模型已保存到 {save_path} ({model_info})")
    
    def load_models(self, path='models', load_dl=False):
        # 从彩票类型对应的子目录加载
        load_path = os.path.join(path, self.lottery_type)
        
        with open(f'{load_path}/rf_front.pkl', 'rb') as f:
            self.rf_models_front = pickle.load(f)
        with open(f'{load_path}/rf_back.pkl', 'rb') as f:
            self.rf_models_back = pickle.load(f)
        with open(f'{load_path}/xgb_front.pkl', 'rb') as f:
            self.xgb_models_front = pickle.load(f)
        with open(f'{load_path}/xgb_back.pkl', 'rb') as f:
            self.xgb_models_back = pickle.load(f)
        with open(f'{load_path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{load_path}/label_encoders_front.pkl', 'rb') as f:
            self.label_encoders_front = pickle.load(f)
        with open(f'{load_path}/label_encoders_back.pkl', 'rb') as f:
            self.label_encoders_back = pickle.load(f)

        # 加载训练时的特征列顺序（旧模型可能没有该文件）
        if os.path.exists(f'{load_path}/feature_cols.pkl'):
            with open(f'{load_path}/feature_cols.pkl', 'rb') as f:
                self.feature_cols = pickle.load(f)
        
        if os.path.exists(f'{load_path}/ensemble_weights.pkl'):
            with open(f'{load_path}/ensemble_weights.pkl', 'rb') as f:
                weights = pickle.load(f)
                # 缺键时回退到 __init__ 默认权重（RF=0.4/XGB=0.6/LSTM=0/Transformer=0），
                # 保持与初始化一致，避免旧权重文件缺键导致 DL 权重被误置为非零
                missing_keys = [k for k in ('rf_weight', 'xgb_weight', 'lstm_weight', 'transformer_weight')
                                if k not in weights]
                if missing_keys:
                    print(f"⚠️  集成权重文件缺少键 {missing_keys}，对应项回退为默认值")
                self.rf_weight = weights.get('rf_weight', 0.4)
                self.xgb_weight = weights.get('xgb_weight', 0.6)
                self.lstm_weight = weights.get('lstm_weight', 0.0)
                self.transformer_weight = weights.get('transformer_weight', 0.0)
            print(f"✅ 模型已从 {path} 加载 (RF={self.rf_weight}, XGB={self.xgb_weight}, LSTM={self.lstm_weight}, Transformer={self.transformer_weight})")
        else:
            print(f"✅ 模型已从 {path} 加载 (使用默认权重)")
        
        if load_dl and os.path.exists(f'{load_path}/dl_config.pkl'):
            try:
                from deep_learning_models import DeepLearningPredictor
                self.dl_predictor = DeepLearningPredictor()
                if self.dl_predictor.load_models(load_path):
                    self.use_deep_learning = True
                else:
                    self.use_deep_learning = False
            except Exception as e:
                print(f"⚠️  深度学习模型加载失败: {e}")
                self.use_deep_learning = False
        else:
            self.use_deep_learning = False

if __name__ == "__main__":
    import pandas as pd
    from db_utils import get_connection
    from feature_engineering import FeatureEngineer

    # 彩种中文名，仅用于日志展示
    TYPE_NAMES = {'dlt': '大乐透', 'ssq': '双色球'}

    def _train_one(lottery_type):
        """训练单个彩种的 RF/XGB/Stacking 模型并保存。

        独立函数 + 调用侧 try/except 隔离：某个彩种数据不足或训练报错时，
        只跳过该彩种（返回 False），不影响其余彩种继续训练。
        """
        name = TYPE_NAMES.get(lottery_type, lottery_type)
        print("=" * 60)
        print(f"开始训练{name} ({lottery_type}) 模型")
        print("=" * 60)

        with get_connection(commit=False) as conn:
            df = pd.read_sql(
                "SELECT * FROM lottery_records WHERE lottery_type=? ORDER BY period",
                conn, params=(lottery_type,)
            )

        if df.empty:
            print(f"❌ 没有找到{name}数据，跳过")
            return False

        fe = FeatureEngineer(df, lottery_type=lottery_type)
        df_features = fe.extract_all_features()
        print(f"✅ 特征提取完成，数据维度: {df_features.shape}")

        predictor = LotteryPredictor(lottery_type=lottery_type)
        predictor.train_random_forest(df_features)
        predictor.train_xgboost(df_features)
        predictor.train_stacking(df_features)
        predictor.save_models(path='models')

        print(f"\n✅ {name}模型训练和保存完成")
        return True

    results = {}
    for lt in ('dlt', 'ssq'):
        try:
            results[lt] = _train_one(lt)
        except Exception as e:
            # 单彩种失败（如数据不足抛 ValueError）不应中断其余彩种训练
            results[lt] = False
            print(f"\n⚠️ {TYPE_NAMES.get(lt, lt)} ({lt}) 训练失败，已跳过：{e}")
        print()

    ok = [lt for lt, v in results.items() if v]
    failed = [lt for lt, v in results.items() if not v]
    print("🎉 模型训练结束！")
    print(f"   成功: {', '.join(ok) if ok else '无'}")
    if failed:
        print(f"   跳过/失败: {', '.join(failed)}")
