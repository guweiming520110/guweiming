"""快速测试Phase 4新特征"""
from database import Database
from feature_engineering import FeatureEngineer

print("🚀 测试Phase 4时间动态特征...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载数据: {len(df)} 条")

print("\n⏳ 提取特征（使用最后500条数据）...")
df_subset = df.tail(500)

fe = FeatureEngineer(df_subset)
df_features = fe.extract_all_features()

print(f"✅ 特征提取完成")
print(f"📈 总特征数: {len(df_features.columns)} 个")

print("\n📊 Phase 4新增特征（13个）:")
new_features = [col for col in df_features.columns if any(x in col for x in ['momentum', 'gap', 'overlap', 'transition'])]
for i, feat in enumerate(sorted(new_features), 1):
    print(f"  {i}. {feat}")

print(f"\n🔍 新特征统计:")
for feat in sorted(new_features)[:5]:
    print(f"\n{feat}:")
    print(f"  均值: {df_features[feat].mean():.4f}")
    print(f"  标准差: {df_features[feat].std():.4f}")
    print(f"  范围: [{df_features[feat].min():.2f}, {df_features[feat].max():.2f}]")

print(f"\n✅ Phase 4特征测试完成！")
print(f"\n特征总结:")
print(f"  - Phase 1基础特征: 11个")
print(f"  - Phase 1高级特征: 11个 (22总计)")
print(f"  - Phase 4时间动态: 13个")
print(f"  - 总计: {len(df_features.columns)} 个特征")
