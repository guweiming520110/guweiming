"""快速评估系统 - 生成首次报告"""
import numpy as np
import pandas as pd
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
from evaluation import LotteryEvaluator
import json

print("🚀 启动快速评估系统...")

db = Database()
df = db.get_all_records()
print(f"✅ 加载 {len(df)} 条历史数据")

if len(df) < 150:
    print(f"❌ 数据量不足：需要至少150条数据，当前只有{len(df)}条")
    exit(1)

fe = FeatureEngineer(df)
df_features = fe.extract_all_features()
print(f"✅ 特征提取完成：{df_features.shape}")

def predictor_func(train_df):
    """简化的预测函数"""
    predictor = LotteryPredictor()
    predictor.train_random_forest(train_df)
    predictor.train_xgboost(train_df)
    
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
    
    latest_features = train_df[available_cols].iloc[-1:].values
    
    result = predictor.predict_with_probabilities(latest_features, play_type='basic')
    pred = result['predictions'][0]
    
    return {
        'front': pred['front'],
        'back': pred['back']
    }

evaluator = LotteryEvaluator()

print("📊 开始Walk-forward验证（快速模式：2个窗口）...")
results = evaluator.walk_forward_validation(
    df_features, 
    predictor_func, 
    train_size=100, 
    test_size=10, 
    step=50
)

predictions_df = pd.DataFrame(results)
print(f"✅ 评估完成：{len(predictions_df)}个预测样本")

print("\n📈 生成评估报告...")
report = evaluator.generate_evaluation_report(predictions_df)

evaluator.print_report(report)

with open('evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 评估报告已保存到 evaluation_report.json")
print(f"\n💡 提示：这是快速评估版本。完整评估请运行: python evaluation.py")
