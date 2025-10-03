"""快速训练深度学习模型（减少训练轮数以加快速度）"""
import sys
sys.path.insert(0, '.')

from deep_learning_models import DeepLearningPredictor
from database import Database
from feature_engineering import FeatureEngineer

print("🚀 快速深度学习模型训练（10 epochs）...")
print("💡 如需完整训练（30 epochs），请稍后运行: python deep_learning_models.py\n")

print("📊 加载历史数据...")
db = Database()
df = db.get_all_records()
print(f"✅ 获取到 {len(df)} 条历史数据")

print("🔧 提取特征...")
fe = FeatureEngineer(df)
df_features = fe.extract_all_features()
print(f"✅ 特征提取完成，数据维度: {df_features.shape}")

print("\n🤖 初始化深度学习模型...")
dl_predictor = DeepLearningPredictor(sequence_length=10)

print("🏋️  开始训练（10 epochs，预计5-8分钟）...\n")
dl_predictor.train_models(df_features, epochs=10, verbose=1)

print("\n💾 保存模型...")
dl_predictor.save_models()

print("\n✅ 快速训练完成！")
print("📊 模型性能预期：")
print("   - LSTM: 捕捉时间序列依赖")
print("   - Transformer: 自注意力机制建模长期依赖")
print("   - 集成权重: RF(20%) + XGBoost(20%) + LSTM(30%) + Transformer(30%)")
print("\n🎯 重启服务器后，系统将自动加载深度学习模型！")
