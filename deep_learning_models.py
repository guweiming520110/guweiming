import numpy as np
import pandas as pd
import pickle
import os

# 深度学习必须显式设置随机种子，否则每次训练结果都不可复现
# （RF/XGBoost 侧已有 random_state=42，这里保持一致）
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
try:
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)
except Exception:  # 未安装/导入失败时至少保证 numpy 侧可复现
    tf = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # 允许在没有 tensorflow 的环境下导入本模块
    keras = None
    layers = None


# ==================== Windows 非 ASCII 路径兼容（TF 2.x 缺陷规避）====================
# TF 的 file_io 在处理含中文等非 ASCII 字符的绝对路径时会抛 UnicodeDecodeError
# （如 G:\数字彩票分析系统\...）。这里提供临时 ASCII 目录中转：
# 保存 → 写到 tempdir(ASCII) → 拷回真实路径；加载 → 复制到 tempdir 再 load。
import shutil as _shutil
import tempfile as _tempfile


def _is_ascii(s):
    try:
        str(s).encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


class _ascii_staging:
    """上下文管理器：为含非 ASCII 的路径提供纯 ASCII 临时目录。"""

    def __init__(self, path):
        self.orig = str(path)
        # 用 C:/temp（若存在）或系统临时目录，保证 ASCII
        base = 'C:/temp' if os.path.isdir('C:/temp') else _tempfile.gettempdir()
        self.target = _tempfile.mkdtemp(prefix='dl_models_', dir=base)

    def commit(self, real_path):
        """把临时目录内容拷回真实路径（保存用）。"""
        os.makedirs(real_path, exist_ok=True)
        for name in os.listdir(self.target):
            _shutil.copy2(os.path.join(self.target, name), os.path.join(real_path, name))
        _shutil.rmtree(self.target, ignore_errors=True)

    def cleanup(self):
        _shutil.rmtree(self.target, ignore_errors=True)


def _load_from_staging(path):
    """若路径含非 ASCII，复制到临时 ASCII 目录并返回临时路径（用完需 cleanup）。

    返回 (实际加载路径, staging对象或None)。
    """
    if _is_ascii(path):
        return path, None
    st = _ascii_staging(path)
    try:
        for name in os.listdir(path):
            _shutil.copy2(os.path.join(path, name), os.path.join(st.target, name))
    except OSError:
        pass
    return st.target, st


class DeepLearningPredictor:
    def __init__(self, sequence_length=10, lottery_type='dlt'):
        self.sequence_length = sequence_length
        self.lottery_type = lottery_type
        self.lstm_models_front = []
        self.lstm_models_back = []
        self.transformer_models_front = []
        self.transformer_models_back = []

        # 根据彩票类型设置号码范围
        self._set_ranges()

    def _set_ranges(self):
        """按彩种重建号码范围。

        S1 修复: load_models 恢复 lottery_type 后必须重建 front_max/back_max，
        否则 ssq 加载时仍用默认 dlt 的 35/12，概率遍历范围错误污染融合。
        """
        if self.lottery_type == 'dlt':
            self.front_max = 35
            self.back_max = 12
        elif self.lottery_type == 'ssq':
            self.front_max = 33
            self.back_max = 16
        else:
            self.front_max = 35
            self.back_max = 12

    def _norm_num(self, num, max_val):
        """号码归一化到 [0,1]（M4: 原始号码 1-max 直接喂网络值域跨度过大，训练不稳定）。"""
        if max_val <= 1:
            return 0.0
        return (float(num) - 1.0) / (max_val - 1.0)

    def prepare_sequence_data(self, df):
        """准备时间序列数据（滑动窗口）

        M4 修复: 输入序列号码归一化到 [0,1]（按各自彩种 max 值），标签保持原始号码。
        """
        df = df.sort_values('period').reset_index(drop=True)

        front_sequences = []
        back_sequences = []
        front_targets = []
        back_targets = []

        for i in range(len(df) - self.sequence_length):
            front_seq = []
            back_seq = []

            for j in range(i, i + self.sequence_length):
                if self.lottery_type == 'dlt':
                    front_nums = [self._norm_num(df.iloc[j][f'front_{k}'], self.front_max) for k in range(1, 6)]
                    back_nums = [self._norm_num(df.iloc[j][f'back_{k}'], self.back_max) for k in range(1, 3)]
                elif self.lottery_type == 'ssq':
                    front_nums = [self._norm_num(df.iloc[j][f'front_{k}'], self.front_max) for k in range(1, 7)]
                    back_nums = [self._norm_num(df.iloc[j][f'back_{1}'], self.back_max)]
                else:
                    front_nums = [self._norm_num(df.iloc[j][f'front_{k}'], self.front_max) for k in range(1, 6)]
                    back_nums = [self._norm_num(df.iloc[j][f'back_{k}'], self.back_max) for k in range(1, 3)]
                front_seq.append(front_nums)
                back_seq.append(back_nums)

            front_sequences.append(front_seq)
            back_sequences.append(back_seq)

            target_row = df.iloc[i + self.sequence_length]
            if self.lottery_type == 'dlt':
                front_targets.append([target_row[f'front_{k}'] for k in range(1, 6)])
                back_targets.append([target_row[f'back_{k}'] for k in range(1, 3)])
            elif self.lottery_type == 'ssq':
                front_targets.append([target_row[f'front_{k}'] for k in range(1, 7)])
                back_targets.append([target_row[f'back_{1}']])
            else:
                front_targets.append([target_row[f'front_{k}'] for k in range(1, 6)])
                back_targets.append([target_row[f'back_{k}'] for k in range(1, 3)])

        return (np.array(front_sequences).astype(np.float32), np.array(back_sequences).astype(np.float32),
                np.array(front_targets), np.array(back_targets))

    def build_lstm_model(self, input_shape, _output_size, max_value):
        """构建双向LSTM模型"""
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(128, return_sequences=True),
                                input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(max_value, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_transformer_model(self, input_shape, _output_size, max_value):
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
        outputs = layers.Dense(max_value, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _fit_model(self, model, X, y, epochs, batch_size, verbose):
        """统一的"时序安全"训练入口。

        keras 的 fit 默认 shuffle=True，会把滑动窗口样本顺序完全打乱；
        更严重的是 validation_split 是在**打乱之后**随机抽取，等价于用
        "未来"验证"过去"，验证指标完全虚高、不可信。
        这里改为：显式 shuffle=False + 手工按时间顺序切最后 10% 作验证集。

        H7 修复: 原切分训练/验证相邻(训练 X[:-val_size]，验证 X[-val_size:])，
        验证首序列(索引 n_val2=n-val_size)的内容行 [n_val2, n_val2+seq_len-1]
        与训练末序列(索引 n_val2-1)的内容行 [n_val2-1, n_val2+seq_len-2]
        重叠 seq_len-1 行——等价于"验证序列首元素恰好出现在最后一条训练样本
        的标签/窗口内"，验证指标轻微乐观。
        现改为训练用 [0, n_val2)，验证用 [n_val2 + seq_len, n)：两段之间保留
        seq_len 的 gap，保证验证序列的任意内容行/标签行都不与训练序列相交。
        """
        n = len(X)
        seq_len = self.sequence_length
        val_size = int(n * 0.1) if n >= 20 else 0
        n_val2 = n - val_size
        if val_size and n_val2 >= 1 and n_val2 + seq_len < n:
            X_train, y_train = X[:n_val2], y[:n_val2]
            validation_data = (X[n_val2 + seq_len:], y[n_val2 + seq_len:])
        else:
            # val_size 不足以覆盖 seq_len 的 gap 时退化为全量训练(无验证集)，
            # 避免负分片；数据量变化最小（仅少 seq_len 个验证样本）。
            X_train, y_train = X, y
            validation_data = None

        model.fit(
            X_train.astype(np.float32), y_train,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose if verbose in (0, 1, 2) else 1,
            shuffle=False,
            validation_data=validation_data,
        )
        return model

    def train_models(self, df, epochs=50, batch_size=32, verbose=0):
        """训练LSTM和Transformer模型

        M3 修复: 返回布尔成功标志，数据不足/无有效序列时返回 False，
        调用方据此决定是否保存模型，避免写出空模型目录并污染集成权重。
        """
        print(f"开始训练深度学习模型，数据量: {len(df)}条")

        front_seq, back_seq, front_targets, back_targets = self.prepare_sequence_data(df)

        if len(front_seq) < self.sequence_length or len(front_seq) < 20:
            print(f"警告：数据不足以训练深度学习模型（需要至少{max(self.sequence_length, 20)}条，当前{len(front_seq)}条序列）")
            return False

        print(f"训练集大小: {len(front_seq)}条序列")

        # 根据彩票类型确定前区数量
        if self.lottery_type == 'dlt':
            num_front = 5
        elif self.lottery_type == 'ssq':
            num_front = 6
        else:
            num_front = 5

        # 判断后区数量
        if len(back_targets) == 0:
            print("警告：无有效后区训练目标，跳过后区模型训练")
            num_back = 0
        else:
            num_back = back_targets.shape[1] if back_targets.ndim > 1 else 1

        for i in range(num_front):
            print(f"  训练前区LSTM模型 {i+1}/{num_front}...")
            y = front_targets[:, i] - 1
            y = y.astype(np.int32)

            lstm_model = self.build_lstm_model(
                (self.sequence_length, num_front), num_front, self.front_max
            )
            self._fit_model(lstm_model, front_seq, y, epochs, batch_size, verbose)
            self.lstm_models_front.append(lstm_model)

            print(f"  训练前区Transformer模型 {i+1}/{num_front}...")
            transformer_model = self.build_transformer_model(
                (self.sequence_length, num_front), num_front, self.front_max
            )
            self._fit_model(transformer_model, front_seq, y, epochs, batch_size, verbose)
            self.transformer_models_front.append(transformer_model)

        for i in range(num_back):
            print(f"  训练后区LSTM模型 {i+1}/{num_back}...")
            # back_targets 始终为 2D (N, num_back)，统一按列取标签，保证 y 为一维 (N,)
            y = back_targets[:, i] - 1
            y = y.astype(np.int32)

            lstm_model = self.build_lstm_model(
                (self.sequence_length, num_back), num_back, self.back_max
            )
            self._fit_model(lstm_model, back_seq, y, epochs, batch_size, verbose)
            self.lstm_models_back.append(lstm_model)

            print(f"  训练后区Transformer模型 {i+1}/{num_back}...")
            transformer_model = self.build_transformer_model(
                (self.sequence_length, num_back), num_back, self.back_max
            )
            self._fit_model(transformer_model, back_seq, y, epochs, batch_size, verbose)
            self.transformer_models_back.append(transformer_model)

        print("✅ 深度学习模型训练完成")
        return True

    def predict_with_probabilities(self, df):
        """使用深度学习模型预测（返回概率）"""
        df = df.sort_values('period').reset_index(drop=True)

        # 数据不足一个序列窗口时直接返回空结果并告警，避免 range 产生负索引回绕取到错误历史
        if len(df) < self.sequence_length:
            print(f"⚠️  数据不足以进行深度学习预测（需要至少{self.sequence_length}条，当前{len(df)}条），跳过DL预测")
            return {
                'front_lstm': {},
                'front_transformer': {},
                'back_lstm': {},
                'back_transformer': {}
            }

        latest_front_seq = []
        latest_back_seq = []

        # 取最近 sequence_length 期作为预测窗口，用切片而非负索引 range 防止回绕
        window = df.iloc[len(df) - self.sequence_length:]
        for _, row in window.iterrows():
            if self.lottery_type == 'dlt':
                # M4: 与训练侧一致，输入号码归一化到 [0,1]
                front_nums = [self._norm_num(row[f'front_{k}'], self.front_max) for k in range(1, 6)]
                back_nums = [self._norm_num(row[f'back_{k}'], self.back_max) for k in range(1, 3)]
            elif self.lottery_type == 'ssq':
                front_nums = [self._norm_num(row[f'front_{k}'], self.front_max) for k in range(1, 7)]
                back_nums = [self._norm_num(row[f'back_{1}'], self.back_max)]
            else:
                front_nums = [self._norm_num(row[f'front_{k}'], self.front_max) for k in range(1, 6)]
                back_nums = [self._norm_num(row[f'back_{k}'], self.back_max) for k in range(1, 3)]
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
        """保存深度学习模型

        Windows 兼容修复: TF 2.x 的 file_io 无法处理含非 ASCII 字符的路径
        （如 G:\数字彩票分析系统\...），会抛 UnicodeDecodeError。
        若目标路径非 ASCII，先保存到系统临时 ASCII 目录再拷回，保证可用。
        """
        # 非 ASCII 路径时用临时目录中转
        tmp_ctx = _ascii_staging(path) if not _is_ascii(path) else None
        target = tmp_ctx.target if tmp_ctx else path
        os.makedirs(target, exist_ok=True)

        try:
            for i, model in enumerate(self.lstm_models_front):
                model.save(f'{target}/lstm_front_{i}.h5')
            for i, model in enumerate(self.lstm_models_back):
                model.save(f'{target}/lstm_back_{i}.h5')
            for i, model in enumerate(self.transformer_models_front):
                model.save(f'{target}/transformer_front_{i}.h5')
            for i, model in enumerate(self.transformer_models_back):
                model.save(f'{target}/transformer_back_{i}.h5')

            with open(f'{target}/dl_config.pkl', 'wb') as f:
                pickle.dump({'sequence_length': self.sequence_length, 'lottery_type': self.lottery_type}, f)

            # 更新集成权重：ml_models.py 训练 RF/XGB 时会把 lstm/transformer 权重写成 0
            # （因深度学习需单独训练）。这里训练完成后回填非零权重，否则加载时深度学习模型
            # 虽被加载(use_deep_learning=True)，却因权重=0 而对融合结果零贡献，等于白训。
            self._update_ensemble_weights(target)
        finally:
            if tmp_ctx:
                tmp_ctx.commit(path)  # 把临时目录产物拷回真实路径

        print(f"✅ 深度学习模型已保存到 {path}")

    def _update_ensemble_weights(self, path):
        """将深度学习权重回填进 ensemble_weights.pkl 并重新归一化四模型权重。

        策略：保留 RF/XGB 的相对比例，把深度学习(LSTM+Transformer)整体占比设为
        DL_SHARE，四者权重之和归一化为 1，使 LSTM/Transformer 真正参与融合。
        若 ensemble_weights.pkl 不存在(未先训练 RF/XGB)，则用默认均衡权重。
        """
        DL_SHARE = 0.4          # 深度学习(LSTM+Transformer)整体占比
        weights_path = f'{path}/ensemble_weights.pkl'

        rf_w, xgb_w = 0.4, 0.6  # 默认 RF/XGB 相对比例(与 ml_models 一致)
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'rb') as f:
                    old = pickle.load(f)
                rf_w = old.get('rf_weight', 0.4)
                xgb_w = old.get('xgb_weight', 0.6)
            except Exception:
                pass

        # 传统模型(RF+XGB)占 1-DL_SHARE，按原相对比例分配
        ml_total = rf_w + xgb_w
        if ml_total <= 0:
            rf_w, xgb_w, ml_total = 0.4, 0.6, 1.0
        ml_share = 1.0 - DL_SHARE
        new_weights = {
            'rf_weight': round(rf_w / ml_total * ml_share, 4),
            'xgb_weight': round(xgb_w / ml_total * ml_share, 4),
            'lstm_weight': round(DL_SHARE / 2, 4),
            'transformer_weight': round(DL_SHARE / 2, 4),
        }
        with open(weights_path, 'wb') as f:
            pickle.dump(new_weights, f)
        print(f"  🔧 已回填集成权重: RF={new_weights['rf_weight']}, "
              f"XGB={new_weights['xgb_weight']}, LSTM={new_weights['lstm_weight']}, "
              f"Transformer={new_weights['transformer_weight']}")

    def load_models(self, path='models'):
        """加载深度学习模型

        Windows 兼容修复: 非 ASCII 路径时先复制到临时 ASCII 目录再 load。
        """
        if not os.path.exists(f'{path}/dl_config.pkl'):
            print("未找到深度学习模型配置")
            return False

        # 非 ASCII 路径中转（用绝对路径判断——相对路径看似 ASCII，TF 内部仍会转成中文绝对路径）
        abs_path = os.path.abspath(path)
        load_path, staging = _load_from_staging(abs_path) if not _is_ascii(abs_path) else (path, None)
        try:
            with open(f'{load_path}/dl_config.pkl', 'rb') as f:
                config = pickle.load(f)
                self.sequence_length = config['sequence_length']
                if 'lottery_type' in config:
                    self.lottery_type = config['lottery_type']
            # S1 修复: 恢复彩种后重建号码范围，避免沿用 dlt 默认值
            self._set_ranges()

            self.lstm_models_front = []
            self.lstm_models_back = []
            self.transformer_models_front = []
            self.transformer_models_back = []

            num_front = 5 if self.lottery_type == 'dlt' else 6
            num_back = 2 if self.lottery_type == 'dlt' else 1

            for i in range(num_front):
                if os.path.exists(f'{load_path}/lstm_front_{i}.h5'):
                    self.lstm_models_front.append(keras.models.load_model(f'{load_path}/lstm_front_{i}.h5'))

            for i in range(num_back):
                if os.path.exists(f'{load_path}/lstm_back_{i}.h5'):
                    self.lstm_models_back.append(keras.models.load_model(f'{load_path}/lstm_back_{i}.h5'))

            for i in range(num_front):
                if os.path.exists(f'{load_path}/transformer_front_{i}.h5'):
                    self.transformer_models_front.append(keras.models.load_model(f'{load_path}/transformer_front_{i}.h5'))

            for i in range(num_back):
                if os.path.exists(f'{load_path}/transformer_back_{i}.h5'):
                    self.transformer_models_back.append(keras.models.load_model(f'{load_path}/transformer_back_{i}.h5'))
        finally:
            if staging:
                staging.cleanup()

        if self.lstm_models_front and self.transformer_models_front:
            print(f"✅ 深度学习模型已加载 (序列长度={self.sequence_length}, 彩种={self.lottery_type})")
            return True
        else:
            print("未找到深度学习模型文件")
            return False


if __name__ == "__main__":
    print("🚀 开始深度学习模型训练...")

    import pandas as pd
    from db_utils import get_connection
    from feature_engineering import FeatureEngineer
    # 注意：模块顶部已 import os，此处不再重复导入

    # 训练大乐透
    print("📊 加载大乐透历史数据...")
    with get_connection(commit=False) as conn:
        df = pd.read_sql(
            "SELECT * FROM lottery_records WHERE lottery_type='dlt' ORDER BY period", conn
        )

    if len(df) > 20:
        print("🔧 提取特征...")
        fe = FeatureEngineer(df, lottery_type='dlt')
        df = fe.extract_all_features()
        print(f"✅ 特征提取完成，数据维度: {df.shape}")

        print("🤖 初始化深度学习模型...")
        dl_predictor = DeepLearningPredictor(sequence_length=10, lottery_type='dlt')

        print("🏋️  开始训练深度学习模型（这将需要10-20分钟）...")
        trained = dl_predictor.train_models(df, epochs=30, verbose=1)

        if trained:
            print("💾 保存模型...")
            dl_predictor.save_models(path=f'models/dlt')

            print("🔮 测试预测...")
            probs = dl_predictor.predict_with_probabilities(df)
            print("\n预测概率示例:")
            print("前区LSTM Top5:", sorted(probs['front_lstm'].items(),
                                          key=lambda x: x[1], reverse=True)[:5])
            print("前区Transformer Top5:", sorted(probs['front_transformer'].items(),
                                                 key=lambda x: x[1], reverse=True)[:5])
            print("\n✅ 大乐透深度学习训练完成！")
        else:
            print("❌ 深度学习训练未成功（数据不足），跳过保存以免写空模型")
    else:
        print(f"❌ 数据不足，需要至少20条历史数据，当前只有{len(df)}条")

    # 训练双色球
    print("\n📊 加载双色球历史数据...")
    with get_connection(commit=False) as conn:
        df_ssq = pd.read_sql(
            "SELECT * FROM lottery_records WHERE lottery_type='ssq' ORDER BY period", conn
        )

    if len(df_ssq) > 20:
        print("🔧 提取特征...")
        fe = FeatureEngineer(df_ssq, lottery_type='ssq')
        df_ssq = fe.extract_all_features()
        print(f"✅ 特征提取完成，数据维度: {df_ssq.shape}")

        print("🤖 初始化深度学习模型...")
        dl_predictor = DeepLearningPredictor(sequence_length=10, lottery_type='ssq')

        print("🏋️  开始训练深度学习模型...")
        trained = dl_predictor.train_models(df_ssq, epochs=30, verbose=1)

        if trained:
            print("💾 保存模型...")
            dl_predictor.save_models(path=f'models/ssq')

            print("🔮 测试预测...")
            probs = dl_predictor.predict_with_probabilities(df_ssq)
            print("\n预测概率示例:")
            print("前区LSTM Top5:", sorted(probs['front_lstm'].items(),
                                          key=lambda x: x[1], reverse=True)[:5])
            print("前区Transformer Top5:", sorted(probs['front_transformer'].items(),
                                                 key=lambda x: x[1], reverse=True)[:5])
            print("\n✅ 双色球深度学习训练完成！")
        else:
            print("❌ 深度学习训练未成功（数据不足），跳过保存以免写空模型")
    else:
        print(f"❌ 数据不足，需要至少20条历史数据，当前只有{len(df_ssq)}条")

    print("\n🎉 所有深度学习模型训练完成！")
