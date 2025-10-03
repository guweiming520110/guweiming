"""最小化评估 - 核心指标，无随机对比"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("🚀 最小化评估系统...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()

train_size = 2700
train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

print(f"📊 训练集: {train_size} | 测试集: {len(test_df)}")

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

print("⏳ 训练RF...")
rf_models = {}
for pos in ['front_1', 'front_2', 'front_3', 'front_4', 'front_5', 'back_1', 'back_2']:
    rf_models[pos] = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    rf_models[pos].fit(X_train_scaled, train_df[pos].values)
print("✅ 完成\n")

results = []
for idx, row in test_df.iterrows():
    features = scaler.transform(row[available_cols].values.reshape(1, -1))
    
    front_pred = [rf_models[f'front_{i}'].predict(features)[0] for i in range(1, 6)]
    back_pred = [rf_models[f'back_{i}'].predict(features)[0] for i in range(1, 3)]
    
    actual_front = set([row['front_1'], row['front_2'], row['front_3'], row['front_4'], row['front_5']])
    actual_back = set([row['back_1'], row['back_2']])
    
    results.append({
        'period': row['period'],
        'front_hits': len(set(front_pred) & actual_front),
        'back_hits': len(set(back_pred) & actual_back)
    })

predictions_df = pd.DataFrame(results)
print(f"✅ {len(predictions_df)}个预测完成\n")

front_avg = predictions_df['front_hits'].mean()
back_avg = predictions_df['back_hits'].mean()
perfect = len(predictions_df[(predictions_df['front_hits'] == 5) & (predictions_df['back_hits'] == 2)])

random_front_expected = 5 * 5 / 35  
random_back_expected = 2 * 2 / 12

print("="*60)
print("📊 真实评估结果（无数据泄漏）")
print("="*60)
print(f"\n🎯 AI预测性能:")
print(f"   前区平均命中: {front_avg:.2f}/5")
print(f"   后区平均命中: {back_avg:.2f}/2")
print(f"   完美预测: {perfect}/{len(predictions_df)}")

print(f"\n📐 理论随机基准:")
print(f"   前区随机期望: {random_front_expected:.2f}/5")
print(f"   后区随机期望: {random_back_expected:.2f}/2")

front_improvement = ((front_avg - random_front_expected) / random_front_expected * 100)
print(f"\n📈 AI vs 随机:")
print(f"   前区提升: {front_improvement:.1f}%")

if front_avg < 1.0:
    print(f"\n💡 结论：AI命中率({front_avg:.2f})接近随机期望({random_front_expected:.2f})")
    print(f"   说明：彩票开奖具有高度随机性，预测能力有限")
elif front_avg < 1.5:
    print(f"\n✨ 结论：AI略优于随机({front_avg:.2f} vs {random_front_expected:.2f})")
else:
    print(f"\n🎉 结论：AI显著优于随机({front_avg:.2f} vs {random_front_expected:.2f})")

print("\n⚠️  免责声明：彩票开奖随机性强，请理性购彩！")
print("="*60)

report = {
    'test_samples': len(predictions_df),
    'ai_performance': {
        'front_avg_hits': front_avg,
        'back_avg_hits': back_avg,
        'perfect_predictions': perfect
    },
    'random_baseline': {
        'front_expected': random_front_expected,
        'back_expected': random_back_expected
    },
    'improvement': {
        'front_percent': front_improvement
    }
}

with open('minimal_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 报告已保存: minimal_evaluation_report.json")
