"""正确的评估系统 - 严格时间序列分割，避免数据泄漏"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
from evaluation import LotteryEvaluator
import json

print("🚀 启动正确评估系统（时间序列分割）...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()

train_size = 2700
test_size = 80

print(f"\n📊 时间序列分割:")
print(f"   训练集: 前 {train_size} 条（期号: {df_features.iloc[0]['period']} - {df_features.iloc[train_size-1]['period']}）")
print(f"   测试集: 后 {test_size} 条（期号: {df_features.iloc[train_size]['period']} - {df_features.iloc[-1]['period']}）")
print(f"   ✅ 严格分离，无数据泄漏\n")

train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

print("⏳ 训练模型（仅使用训练集）...")
predictor = LotteryPredictor()
predictor.train_random_forest(train_df)
predictor.train_xgboost(train_df)
print("✅ 模型训练完成\n")

base_features = ['odd_count', 'even_count', 'large_count', 'small_count', 
                'front_sum', 'ac_value', 'span', 'max_front_omission', 
                'avg_front_omission', 'max_back_omission', 'avg_back_omission']

advanced_features = ['max_pair_freq', 'max_triplet_freq', 'pmi_score',
                   'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
                   'consecutive_count', 'max_consecutive',
                   'zone1_count', 'zone2_count', 'zone3_count',
                   'rolling_std', 'rolling_entropy', 'similarity_score']

feature_cols = base_features + advanced_features
available_cols = [col for col in feature_cols if col in df_features.columns]

print("🔮 在测试集上进行预测...")
results = []

for idx, row in test_df.iterrows():
    features = row[available_cols].values.reshape(1, -1)
    
    result = predictor.predict_with_probabilities(features, play_type='basic')
    pred = result['predictions'][0]
    
    actual_front = set([row['front_1'], row['front_2'], row['front_3'], 
                       row['front_4'], row['front_5']])
    actual_back = set([row['back_1'], row['back_2']])
    
    results.append({
        'period': row['period'],
        'train_start': train_df.iloc[0]['period'],
        'train_end': train_df.iloc[-1]['period'],
        'predicted_front': pred['front'],
        'predicted_back': pred['back'],
        'actual_front': list(actual_front),
        'actual_back': list(actual_back),
        'front_hits': len(set(pred['front']) & actual_front),
        'back_hits': len(set(pred['back']) & actual_back)
    })

predictions_df = pd.DataFrame(results)
print(f"✅ 预测完成：{len(predictions_df)}个样本\n")

evaluator = LotteryEvaluator()
report = evaluator.generate_evaluation_report(predictions_df)

evaluator.print_report(report)

with open('evaluation_report_proper.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 正确评估报告已保存到 evaluation_report_proper.json")

print("\n" + "="*60)
print("💡 关键洞察（无数据泄漏版本）:")
ai_front = report['baseline_comparison']['ai_performance']['front_hits_avg']
random_front = report['baseline_comparison']['random_baseline']['front_hits_avg']
improvement = report['baseline_comparison']['improvement_percent']['front_improvement']

if improvement > 20:
    print(f"   ✨ AI显著优于随机：前区命中({ai_front:.2f})比随机({random_front:.2f})高{improvement:.1f}%")
elif improvement > 5:
    print(f"   👌 AI略优于随机：前区命中({ai_front:.2f})比随机({random_front:.2f})高{improvement:.1f}%")
elif improvement > -5:
    print(f"   📊 AI与随机相近：前区命中({ai_front:.2f})，随机({random_front:.2f})")
    print(f"   💡 说明：彩票开奖具有高度随机性，AI预测能力有限")
else:
    print(f"   ⚠️ AI略差于随机：前区命中({ai_front:.2f})，随机({random_front:.2f})")

print(f"   📈 前区Top-5候选命中率: {report['hit_rate_at_k']['front_hit_rate@5']:.1%}")
print(f"   💰 单注期望收益: ¥{report['expected_value']['expected_value_per_bet']:.2f}")

if report['expected_value']['net_profit'] < 0:
    roi = report['expected_value']['roi_percent']
    print(f"   📉 ROI: {roi:.1f}% (期望为负，符合彩票随机性)")
    print(f"   ⚠️  建议：彩票仅供娱乐，请理性购彩")
else:
    print(f"   📈 ROI: {report['expected_value']['roi_percent']:.1f}%")

print("="*60)

print("\n📝 评估总结:")
print(f"   ✅ 训练集：{train_size}条")
print(f"   ✅ 测试集：{test_size}条（未见数据）")
print(f"   ✅ 前区平均命中：{ai_front:.2f}/5")
print(f"   ✅ 后区平均命中：{report['baseline_comparison']['ai_performance']['back_hits_avg']:.2f}/2")
print(f"   ✅ 完美预测次数：{report['overall_accuracy']['perfect_predictions']}/80")
