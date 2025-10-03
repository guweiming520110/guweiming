"""使用Phase 4新特征重新训练模型"""
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
from feature_config import get_all_features, print_feature_summary
import os

print("🚀 Phase 4模型重训练开始...")

print("\n📊 特征配置:")
total_features = print_feature_summary()

db = Database()
df = db.get_all_records()
print(f"\n✅ 加载数据: {len(df)} 条")

print("\n⏳ 提取特征（包含Phase 4时间动态特征）...")
fe = FeatureEngineer(df)
df_features = fe.extract_all_features()
print(f"✅ 特征提取完成: {df_features.shape}")

features_list = get_all_features()
available_features = [col for col in features_list if col in df_features.columns]
print(f"📈 可用特征: {len(available_features)}/{len(features_list)}")

if os.path.exists('models/rf_front.pkl'):
    print("\n🗑️  删除旧模型...")
    import glob
    for file in glob.glob('models/*.pkl'):
        os.remove(file)
        print(f"   已删除: {file}")

print("\n⏳ 训练新模型（使用44个特征）...")
predictor = LotteryPredictor()

print("\n1️⃣  训练随机森林...")
predictor.train_random_forest(df_features)

print("\n2️⃣  训练XGBoost...")
predictor.train_xgboost(df_features)

print("\n3️⃣  训练Stacking集成...")
predictor.train_stacking(df_features)

print("\n💾 保存模型...")
predictor.save_models()

print("\n✅ Phase 4模型训练完成！")
print("\n📊 模型性能提升预期:")
print("   - 短期动量特征：捕捉热号/冷号趋势")
print("   - 条件间隔特征：识别连续不出现模式")
print("   - 转换特征：利用期间相关性")
print("\n🎯 建议下一步：运行评估对比新旧模型性能")
