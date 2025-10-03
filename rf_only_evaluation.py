"""快速评估 - 仅使用随机森林（避免XGBoost训练耗时）"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("🚀 启动快速RF评估系统...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()

train_size = 2700
test_size = 80

print(f"\n📊 时间序列分割（严格无泄漏）:")
print(f"   训练集: 前 {train_size} 条")
print(f"   测试集: 后 {test_size} 条\n")

train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

base_features = ['odd_count', 'even_count', 'large_count', 'small_count', 
                'front_sum', 'ac_value', 'span', 'max_front_omission', 
                'avg_front_omission', 'max_back_omission', 'avg_back_omission']

advanced_features = ['max_pair_freq', 'max_triplet_freq', 'pmi_score',
                   'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
                   'consecutive_count', 'max_consecutive',
                   'zone1_count', 'zone2_count', 'zone3_count',
                   'rolling_std', 'rolling_entropy', 'similarity_score']

feature_cols = base_features + advanced_features
available_cols = [col for col in feature_cols if col in train_df.columns]

X_train = train_df[available_cols].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("⏳ 训练随机森林模型...")
rf_models = {}
for i, pos in enumerate(['front_1', 'front_2', 'front_3', 'front_4', 'front_5']):
    rf_models[pos] = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_models[pos].fit(X_train_scaled, train_df[pos].values)

for i, pos in enumerate(['back_1', 'back_2']):
    rf_models[pos] = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_models[pos].fit(X_train_scaled, train_df[pos].values)

print("✅ RF模型训练完成\n")

print("🔮 在测试集上预测...")
results = []

for idx, row in test_df.iterrows():
    features = row[available_cols].values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    front_pred = []
    for pos in ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']:
        proba = rf_models[pos].predict_proba(features_scaled)[0]
        top_idx = np.argsort(proba)[-1]
        front_pred.append(rf_models[pos].classes_[top_idx])
    
    back_pred = []
    for pos in ['back_1', 'back_2']:
        proba = rf_models[pos].predict_proba(features_scaled)[0]
        top_idx = np.argsort(proba)[-1]
        back_pred.append(rf_models[pos].classes_[top_idx])
    
    actual_front = set([row['front_1'], row['front_2'], row['front_3'], 
                       row['front_4'], row['front_5']])
    actual_back = set([row['back_1'], row['back_2']])
    
    results.append({
        'period': row['period'],
        'predicted_front': front_pred,
        'predicted_back': back_pred,
        'actual_front': list(actual_front),
        'actual_back': list(actual_back),
        'front_hits': len(set(front_pred) & actual_front),
        'back_hits': len(set(back_pred) & actual_back)
    })

predictions_df = pd.DataFrame(results)
print(f"✅ 预测完成：{len(predictions_df)}个样本\n")

from evaluation import LotteryEvaluator
evaluator = LotteryEvaluator()
report = evaluator.generate_evaluation_report(predictions_df)

evaluator.print_report(report)

with open('rf_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ RF评估报告已保存到 rf_evaluation_report.json")

print("\n" + "="*60)
print("💡 真实评估结果（无数据泄漏）:")
ai_front = report['baseline_comparison']['ai_performance']['front_hits_avg']
random_front = report['baseline_comparison']['random_baseline']['front_hits_avg']
improvement = report['baseline_comparison']['improvement_percent']['front_improvement']

print(f"   🎯 AI前区命中: {ai_front:.2f}/5")
print(f"   🎲 随机命中: {random_front:.2f}/5")
print(f"   📊 提升幅度: {improvement:.1f}%")
print(f"   💰 期望收益: ¥{report['expected_value']['expected_value_per_bet']:.2f}")
print(f"   📈 Hit-rate@5: {report['hit_rate_at_k']['front_hit_rate@5']:.1%}")

if improvement < 5:
    print(f"\n   💡 结论：AI预测效果接近随机，符合彩票高随机性特征")
    print(f"   ⚠️  建议：彩票仅供娱乐，切勿过度投资")
elif improvement < 20:
    print(f"\n   ✨ 结论：AI略优于随机，有一定预测能力")
else:
    print(f"\n   🎉 结论：AI显著优于随机，算法有效！")

print("="*60)
