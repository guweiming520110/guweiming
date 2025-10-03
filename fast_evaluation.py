"""快速评估 - 使用现有模型"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
from evaluation import LotteryEvaluator
import json
import os

print("🚀 启动快速评估系统（使用现有模型）...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()

predictor = LotteryPredictor()

if os.path.exists('models/rf_front.pkl'):
    print("✅ 加载现有模型...")
    predictor.load_models()
    predictor.full_df = df_features.copy()
else:
    print("⚠️ 未找到现有模型，跳过评估")
    print("请先运行预测功能训练模型")
    exit(1)

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

print(f"\n📊 对最近50期开奖进行预测评估...")
test_df = df_features.tail(50)
results = []

for idx, row in test_df.iterrows():
    features = row[available_cols].values.reshape(1, -1)
    
    try:
        result = predictor.predict_with_probabilities(features, play_type='basic')
        pred = result['predictions'][0]
        
        actual_front = set([row['front_1'], row['front_2'], row['front_3'], 
                           row['front_4'], row['front_5']])
        actual_back = set([row['back_1'], row['back_2']])
        
        results.append({
            'period': row['period'],
            'train_start': 'N/A',
            'train_end': 'N/A',
            'predicted_front': pred['front'],
            'predicted_back': pred['back'],
            'actual_front': list(actual_front),
            'actual_back': list(actual_back),
            'front_hits': len(set(pred['front']) & actual_front),
            'back_hits': len(set(pred['back']) & actual_back)
        })
    except Exception as e:
        print(f"预测失败: {e}")
        continue

predictions_df = pd.DataFrame(results)
print(f"✅ 预测完成：{len(predictions_df)}个样本")

evaluator = LotteryEvaluator()
report = evaluator.generate_evaluation_report(predictions_df)

evaluator.print_report(report)

with open('evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 评估报告已保存到 evaluation_report.json")

print("\n" + "="*60)
print("💡 关键洞察:")
ai_front = report['baseline_comparison']['ai_performance']['front_hits_avg']
random_front = report['baseline_comparison']['random_baseline']['front_hits_avg']
improvement = report['baseline_comparison']['improvement_percent']['front_improvement']

if improvement > 5:
    print(f"   ✨ AI性能优异：前区命中({ai_front:.2f})比随机({random_front:.2f})高{improvement:.1f}%")
elif improvement > 0:
    print(f"   👌 AI略优于随机：前区命中({ai_front:.2f})比随机({random_front:.2f})高{improvement:.1f}%")
else:
    print(f"   📊 AI与随机相近：前区命中({ai_front:.2f})，随机({random_front:.2f})")

print(f"   📈 前区Top-5候选命中率: {report['hit_rate_at_k']['front_hit_rate@5']:.1%}")
print(f"   💰 单注期望收益: ¥{report['expected_value']['expected_value_per_bet']:.2f}")

if report['expected_value']['net_profit'] < 0:
    roi = report['expected_value']['roi_percent']
    print(f"   ⚠️  期望收益为负({roi:.1f}%)，符合彩票随机性")
    print(f"   💡 建议：仅作娱乐，理性购彩")

print("="*60)
