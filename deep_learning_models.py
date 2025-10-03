import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

class DeepLearningPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.lstm_models_front = []
        self.lstm_models_back = []
        self.transformer_models_front = []
        self.transformer_models_back = []
        
        self.front_max = 35
        self.back_max = 12
        
    def prepare_sequence_data(self, df):
        """准备时间序列数据（滑动窗口）"""
        df = df.sort_values('period').reset_index(drop=True)
        
        front_sequences = []
        back_sequences = []
        front_targets = []
        back_targets = []
        
        for i in range(len(df) - self.sequence_length):
            front_seq = []
            back_seq = []
            
            for j in range(i, i + self.sequence_length):
                front_nums = [df.iloc[j][f'front_{k}'] for k in range(1, 6)]
                back_nums = [df.iloc[j][f'back_{k}'] for k in range(1, 3)]
                front_seq.append(front_nums)
                back_seq.append(back_nums)
            
            front_sequences.append(front_seq)
            back_sequences.append(back_seq)
            
            target_row = df.iloc[i + self.sequence_length]
            front_targets.append([target_row[f'front_{k}'] for k in range(1, 6)])
            back_targets.append([target_row[f'back_{k}'] for k in range(1, 3)])
        
        return (np.array(front_sequences), np.array(back_sequences),
                np.array(front_targets), np.array(back_targets))
    
    def build_lstm_model(self, input_shape, output_size, max_value):
        """构建双向LSTM模型"""
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(128, return_sequences=True), 
                                input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(max_value + 1, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transformer_model(self, input_shape, output_size, max_value):
        """构建Transformer模型（自注意力机制）"""
        inputs = keras.Input(shape=input_shape)
        
        x = layers.Dense(64)(inputs)
        
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )(x, x)
        
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        ffn = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64)
        ])
        
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(max_value + 1, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, df, epochs=50, batch_size=32, verbose=0):
        """训练LSTM和Transformer模型"""
        print(f"开始训练深度学习模型，数据量: {len(df)}条")
        
        front_seq, back_seq, front_targets, back_targets = self.prepare_sequence_data(df)
        
        if len(front_seq) < self.sequence_length:
            print(f"警告：数据不足以训练深度学习模型（需要至少{self.sequence_length}条）")
            return
        
        print(f"训练集大小: {len(front_seq)}条序列")
        
        for i in range(5):
            print(f"  训练前区LSTM模型 {i+1}/5...")
            y = front_targets[:, i] - 1
            
            lstm_model = self.build_lstm_model(
                (self.sequence_length, 5), 5, self.front_max
            )
            lstm_model.fit(front_seq, y, epochs=epochs, batch_size=batch_size, 
                          verbose=verbose, validation_split=0.1)
            self.lstm_models_front.append(lstm_model)
            
            print(f"  训练前区Transformer模型 {i+1}/5...")
            transformer_model = self.build_transformer_model(
                (self.sequence_length, 5), 5, self.front_max
            )
            transformer_model.fit(front_seq, y, epochs=epochs, batch_size=batch_size,
                                 verbose=verbose, validation_split=0.1)
            self.transformer_models_front.append(transformer_model)
        
        for i in range(2):
            print(f"  训练后区LSTM模型 {i+1}/2...")
            y = back_targets[:, i] - 1
            
            lstm_model = self.build_lstm_model(
                (self.sequence_length, 2), 2, self.back_max
            )
            lstm_model.fit(back_seq, y, epochs=epochs, batch_size=batch_size,
                          verbose=verbose, validation_split=0.1)
            self.lstm_models_back.append(lstm_model)
            
            print(f"  训练后区Transformer模型 {i+1}/2...")
            transformer_model = self.build_transformer_model(
                (self.sequence_length, 2), 2, self.back_max
            )
            transformer_model.fit(back_seq, y, epochs=epochs, batch_size=batch_size,
                                 verbose=verbose, validation_split=0.1)
            self.transformer_models_back.append(transformer_model)
        
        print("✅ 深度学习模型训练完成")
    
    def predict_with_probabilities(self, df):
        """使用深度学习模型预测（返回概率）"""
        df = df.sort_values('period').reset_index(drop=True)
        
        latest_front_seq = []
        latest_back_seq = []
        
        for i in range(len(df) - self.sequence_length, len(df)):
            front_nums = [df.iloc[i][f'front_{k}'] for k in range(1, 6)]
            back_nums = [df.iloc[i][f'back_{k}'] for k in range(1, 3)]
            latest_front_seq.append(front_nums)
            latest_back_seq.append(back_nums)
        
        front_input = np.array([latest_front_seq])
        back_input = np.array([latest_back_seq])
        
        front_lstm_probs = {}
        front_transformer_probs = {}
        back_lstm_probs = {}
        back_transformer_probs = {}
        
        for i, model in enumerate(self.lstm_models_front):
            probs = model.predict(front_input, verbose=0)[0]
            for num in range(1, self.front_max + 1):
                if num not in front_lstm_probs:
                    front_lstm_probs[num] = []
                front_lstm_probs[num].append(probs[num - 1] if num - 1 < len(probs) else 0)
        
        for i, model in enumerate(self.transformer_models_front):
            probs = model.predict(front_input, verbose=0)[0]
            for num in range(1, self.front_max + 1):
                if num not in front_transformer_probs:
                    front_transformer_probs[num] = []
                front_transformer_probs[num].append(probs[num - 1] if num - 1 < len(probs) else 0)
        
        for i, model in enumerate(self.lstm_models_back):
            probs = model.predict(back_input, verbose=0)[0]
            for num in range(1, self.back_max + 1):
                if num not in back_lstm_probs:
                    back_lstm_probs[num] = []
                back_lstm_probs[num].append(probs[num - 1] if num - 1 < len(probs) else 0)
        
        for i, model in enumerate(self.transformer_models_back):
            probs = model.predict(back_input, verbose=0)[0]
            for num in range(1, self.back_max + 1):
                if num not in back_transformer_probs:
                    back_transformer_probs[num] = []
                back_transformer_probs[num].append(probs[num - 1] if num - 1 < len(probs) else 0)
        
        for num in front_lstm_probs:
            front_lstm_probs[num] = np.mean(front_lstm_probs[num])
        for num in front_transformer_probs:
            front_transformer_probs[num] = np.mean(front_transformer_probs[num])
        for num in back_lstm_probs:
            back_lstm_probs[num] = np.mean(back_lstm_probs[num])
        for num in back_transformer_probs:
            back_transformer_probs[num] = np.mean(back_transformer_probs[num])
        
        return {
            'front_lstm': front_lstm_probs,
            'front_transformer': front_transformer_probs,
            'back_lstm': back_lstm_probs,
            'back_transformer': back_transformer_probs
        }
    
    def save_models(self, path='models'):
        """保存深度学习模型"""
        os.makedirs(path, exist_ok=True)
        
        for i, model in enumerate(self.lstm_models_front):
            model.save(f'{path}/lstm_front_{i}.h5')
        for i, model in enumerate(self.lstm_models_back):
            model.save(f'{path}/lstm_back_{i}.h5')
        for i, model in enumerate(self.transformer_models_front):
            model.save(f'{path}/transformer_front_{i}.h5')
        for i, model in enumerate(self.transformer_models_back):
            model.save(f'{path}/transformer_back_{i}.h5')
        
        with open(f'{path}/dl_config.pkl', 'wb') as f:
            pickle.dump({'sequence_length': self.sequence_length}, f)
        
        print(f"✅ 深度学习模型已保存到 {path}")
    
    def load_models(self, path='models'):
        """加载深度学习模型"""
        if not os.path.exists(f'{path}/dl_config.pkl'):
            print("未找到深度学习模型配置")
            return False
        
        with open(f'{path}/dl_config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.sequence_length = config['sequence_length']
        
        self.lstm_models_front = []
        self.lstm_models_back = []
        self.transformer_models_front = []
        self.transformer_models_back = []
        
        for i in range(5):
            if os.path.exists(f'{path}/lstm_front_{i}.h5'):
                self.lstm_models_front.append(keras.models.load_model(f'{path}/lstm_front_{i}.h5'))
        
        for i in range(2):
            if os.path.exists(f'{path}/lstm_back_{i}.h5'):
                self.lstm_models_back.append(keras.models.load_model(f'{path}/lstm_back_{i}.h5'))
        
        for i in range(5):
            if os.path.exists(f'{path}/transformer_front_{i}.h5'):
                self.transformer_models_front.append(keras.models.load_model(f'{path}/transformer_front_{i}.h5'))
        
        for i in range(2):
            if os.path.exists(f'{path}/transformer_back_{i}.h5'):
                self.transformer_models_back.append(keras.models.load_model(f'{path}/transformer_back_{i}.h5'))
        
        if self.lstm_models_front and self.transformer_models_front:
            print(f"✅ 深度学习模型已加载 (序列长度={self.sequence_length})")
            return True
        else:
            print("未找到深度学习模型文件")
            return False

if __name__ == "__main__":
    print("🚀 开始深度学习模型训练...")
    
    from feature_engineering import FeatureEngineer
    import os
    
    print("📊 加载历史数据...")
    csv_path = 'lottery_data.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ 从CSV获取到 {len(df)} 条历史数据")
    else:
        from database import Database
        db = Database()
        df = db.get_all_records()
        print(f"✅ 从数据库获取到 {len(df)} 条历史数据")
    
    if len(df) > 20:
        print("🔧 提取特征...")
        fe = FeatureEngineer(df)
        df = fe.extract_all_features()
        print(f"✅ 特征提取完成，数据维度: {df.shape}")
        
        print("🤖 初始化深度学习模型...")
        dl_predictor = DeepLearningPredictor(sequence_length=10)
        
        print("🏋️  开始训练深度学习模型（这将需要10-20分钟）...")
        dl_predictor.train_models(df, epochs=30, verbose=1)
        
        print("💾 保存模型...")
        dl_predictor.save_models()
        
        print("🔮 测试预测...")
        probs = dl_predictor.predict_with_probabilities(df)
        print("\n预测概率示例:")
        print("前区LSTM Top5:", sorted(probs['front_lstm'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5])
        print("前区Transformer Top5:", sorted(probs['front_transformer'].items(),
                                             key=lambda x: x[1], reverse=True)[:5])
        print("\n✅ 深度学习训练完成！")
    else:
        print(f"❌ 数据不足，需要至少20条历史数据，当前只有{len(df)}条")
