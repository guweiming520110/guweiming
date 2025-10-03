"""演示评估系统 - 单窗口快速报告"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
from evaluation import LotteryEvaluator
import json

print("🚀 启动演示评估系统...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()

predictor = LotteryPredictor()
predictor.train_random_forest(df_features[:2500])
predictor.train_xgboost(df_features[:2500])
print("✅ 模型训练完成（训练集：前2500条）")

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

print(f"\n📊 对最近100期开奖进行预测评估...")
test_df = df_features.tail(100)
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
        'train_start': df_features.iloc[0]['period'],
        'train_end': df_features.iloc[2499]['period'],
        'predicted_front': pred['front'],
        'predicted_back': pred['back'],
        'actual_front': list(actual_front),
        'actual_back': list(actual_back),
        'front_hits': len(set(pred['front']) & actual_front),
        'back_hits': len(set(pred['back']) & actual_back)
    })

predictions_df = pd.DataFrame(results)
print(f"✅ 预测完成：{len(predictions_df)}个样本")

evaluator = LotteryEvaluator()
report = evaluator.generate_evaluation_report(predictions_df)

evaluator.print_report(report)

with open('evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 评估报告已保存到 evaluation_report.json")

print("\n💡 关键发现:")
ai_front = report['baseline_comparison']['ai_performance']['front_hits_avg']
random_front = report['baseline_comparison']['random_baseline']['front_hits_avg']
improvement = report['baseline_comparison']['improvement_percent']['front_improvement']

if improvement > 0:
    print(f"   🎯 AI预测前区命中率({ai_front:.2f})比随机选号({random_front:.2f})高{improvement:.1f}%")
else:
    print(f"   ⚠️ AI预测前区命中率({ai_front:.2f})与随机选号({random_front:.2f})相近")

print(f"   📈 前区Hit-rate@5: {report['hit_rate_at_k']['front_hit_rate@5']:.1%}")
print(f"   💰 单注期望收益: ¥{report['expected_value']['expected_value_per_bet']:.2f}")
