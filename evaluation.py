import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
import json

class LotteryEvaluator:
    """大乐透预测评估系统 - Walk-forward验证 + 多维度指标"""
    
    def __init__(self):
        self.prize_structure = [
            {'name': 'first', 'front': 5, 'back': 2, 'prize': 10000000},
            {'name': 'second', 'front': 5, 'back': 1, 'prize': 500000},
            {'name': 'third', 'front': 5, 'back': 0, 'prize': 10000},
            {'name': 'fourth', 'front': 4, 'back': 2, 'prize': 3000},
            {'name': 'fifth', 'front': 4, 'back': 1, 'prize': 300},
            {'name': 'sixth', 'front': 3, 'back': 2, 'prize': 200},
            {'name': 'sixth', 'front': 4, 'back': 0, 'prize': 200},
            {'name': 'seventh', 'front': 2, 'back': 2, 'prize': 100},
            {'name': 'seventh', 'front': 3, 'back': 1, 'prize': 100},
            {'name': 'eighth', 'front': 1, 'back': 2, 'prize': 15},
            {'name': 'eighth', 'front': 0, 'back': 2, 'prize': 15},
            {'name': 'ninth', 'front': 3, 'back': 0, 'prize': 5},
            {'name': 'ninth', 'front': 2, 'back': 1, 'prize': 5}
        ]
    
    def walk_forward_validation(self, df, predictor_func, train_size=100, test_size=50, step=10):
        """
        Walk-forward时间序列验证（修复版：每期独立预测）
        
        参数:
            df: 历史数据
            predictor_func: 预测函数 func(train_df, test_row) -> predictions
            train_size: 初始训练集大小
            test_size: 每次评估的测试期数
            step: 滑动步长
        
        返回:
            评估结果字典
        """
        df = df.sort_values('period').reset_index(drop=True)
        results = []
        
        total_windows = (len(df) - train_size - test_size) // step + 1
        print(f"🔄 开始Walk-forward验证：{total_windows}个时间窗口")
        
        for i in range(0, len(df) - train_size - test_size + 1, step):
            train_df = df.iloc[i:i+train_size]
            test_df = df.iloc[i+train_size:i+train_size+test_size]
            
            for idx, row in test_df.iterrows():
                predictions = predictor_func(train_df, row)
                
                actual_front = set([row['front_1'], row['front_2'], row['front_3'], 
                                   row['front_4'], row['front_5']])
                actual_back = set([row['back_1'], row['back_2']])
                
                result = {
                    'period': row['period'],
                    'train_start': train_df.iloc[0]['period'],
                    'train_end': train_df.iloc[-1]['period'],
                    'predicted_front': predictions.get('front', []),
                    'predicted_back': predictions.get('back', []),
                    'actual_front': list(actual_front),
                    'actual_back': list(actual_back),
                    'front_hits': len(set(predictions.get('front', [])) & actual_front),
                    'back_hits': len(set(predictions.get('back', [])) & actual_back)
                }
                results.append(result)
        
        return results
    
    def calculate_hit_rate_at_k(self, predictions_df, k_values=[5, 10, 15]):
        """
        计算Hit-rate@k：Top-k候选号码的命中率
        
        参数:
            predictions_df: 预测结果DataFrame
            k_values: k值列表
        
        返回:
            各k值的命中率统计
        """
        hit_rates = {}
        
        for k in k_values:
            front_hits = 0
            back_hits = 0
            total = len(predictions_df)
            
            for _, row in predictions_df.iterrows():
                pred_front = set(row['predicted_front'][:k])
                pred_back = set(row['predicted_back'][:min(k, len(row['predicted_back']))])
                
                actual_front = set(row['actual_front'])
                actual_back = set(row['actual_back'])
                
                if pred_front & actual_front:
                    front_hits += 1
                if pred_back & actual_back:
                    back_hits += 1
            
            hit_rates[f'front_hit_rate@{k}'] = front_hits / total if total > 0 else 0
            hit_rates[f'back_hit_rate@{k}'] = back_hits / total if total > 0 else 0
        
        return hit_rates
    
    def calculate_expected_value(self, predictions_df, bet_cost=2):
        """
        计算期望收益（Expected Value）
        
        参数:
            predictions_df: 预测结果DataFrame
            bet_cost: 单注成本（元）
        
        返回:
            期望收益统计
        """
        total_investment = len(predictions_df) * bet_cost
        total_prize = 0
        prize_distribution = defaultdict(int)
        
        for _, row in predictions_df.iterrows():
            front_hits = row['front_hits']
            back_hits = row['back_hits']
            
            for prize_info in self.prize_structure:
                if (front_hits == prize_info['front'] and 
                    back_hits == prize_info['back']):
                    prize_distribution[prize_info['name']] += 1
                    total_prize += prize_info['prize']
                    break
        
        expected_value = (total_prize - total_investment) / len(predictions_df)
        roi = ((total_prize - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        return {
            'total_investment': total_investment,
            'total_prize': total_prize,
            'net_profit': total_prize - total_investment,
            'expected_value_per_bet': expected_value,
            'roi_percent': roi,
            'prize_distribution': dict(prize_distribution)
        }
    
    def calculate_per_position_accuracy(self, predictions_df):
        """
        计算每个位置的预测准确率（修复版：匹配排序后的位置）
        
        返回:
            各位置准确率统计
        """
        position_stats = {
            'front_positions': {i: {'correct': 0, 'total': 0} for i in range(1, 6)},
            'back_positions': {i: {'correct': 0, 'total': 0} for i in range(1, 3)}
        }
        
        for _, row in predictions_df.iterrows():
            pred_front = sorted(row['predicted_front'][:5])
            pred_back = sorted(row['predicted_back'][:2])
            actual_front = sorted(row['actual_front'])
            actual_back = sorted(row['actual_back'])
            
            for i in range(min(len(pred_front), len(actual_front))):
                position_stats['front_positions'][i+1]['total'] += 1
                if i < len(pred_front) and i < len(actual_front) and pred_front[i] == actual_front[i]:
                    position_stats['front_positions'][i+1]['correct'] += 1
            
            for i in range(min(len(pred_back), len(actual_back))):
                position_stats['back_positions'][i+1]['total'] += 1
                if i < len(pred_back) and i < len(actual_back) and pred_back[i] == actual_back[i]:
                    position_stats['back_positions'][i+1]['correct'] += 1
        
        position_accuracy = {
            'front': {},
            'back': {}
        }
        
        for i, stats in position_stats['front_positions'].items():
            if stats['total'] > 0:
                position_accuracy['front'][f'position_{i}'] = stats['correct'] / stats['total']
        
        for i, stats in position_stats['back_positions'].items():
            if stats['total'] > 0:
                position_accuracy['back'][f'position_{i}'] = stats['correct'] / stats['total']
        
        return position_accuracy
    
    def random_baseline_comparison(self, predictions_df, n_simulations=100):
        """
        与随机选号基准对比
        
        参数:
            predictions_df: 预测结果DataFrame
            n_simulations: 随机模拟次数
        
        返回:
            AI vs 随机对比结果
        """
        ai_front_hits = predictions_df['front_hits'].mean()
        ai_back_hits = predictions_df['back_hits'].mean()
        
        random_front_hits_list = []
        random_back_hits_list = []
        
        for _ in range(n_simulations):
            random_front = set(np.random.choice(range(1, 36), 5, replace=False))
            random_back = set(np.random.choice(range(1, 13), 2, replace=False))
            
            sim_front_hits = 0
            sim_back_hits = 0
            
            for _, row in predictions_df.iterrows():
                actual_front = set(row['actual_front'])
                actual_back = set(row['actual_back'])
                
                sim_front_hits += len(random_front & actual_front)
                sim_back_hits += len(random_back & actual_back)
            
            random_front_hits_list.append(sim_front_hits / len(predictions_df))
            random_back_hits_list.append(sim_back_hits / len(predictions_df))
        
        random_front_mean = np.mean(random_front_hits_list)
        random_back_mean = np.mean(random_back_hits_list)
        
        improvement = {
            'front_improvement': ((ai_front_hits - random_front_mean) / random_front_mean * 100) 
                                if random_front_mean > 0 else 0,
            'back_improvement': ((ai_back_hits - random_back_mean) / random_back_mean * 100) 
                               if random_back_mean > 0 else 0
        }
        
        return {
            'ai_performance': {
                'front_hits_avg': ai_front_hits,
                'back_hits_avg': ai_back_hits
            },
            'random_baseline': {
                'front_hits_avg': random_front_mean,
                'front_hits_std': np.std(random_front_hits_list),
                'back_hits_avg': random_back_mean,
                'back_hits_std': np.std(random_back_hits_list)
            },
            'improvement_percent': improvement
        }
    
    def generate_evaluation_report(self, predictions_df):
        """
        生成完整评估报告
        
        返回:
            综合评估报告字典
        """
        print("📊 生成评估报告...")
        
        hit_rates = self.calculate_hit_rate_at_k(predictions_df, k_values=[5, 10, 15])
        expected_value = self.calculate_expected_value(predictions_df)
        position_accuracy = self.calculate_per_position_accuracy(predictions_df)
        baseline_comparison = self.random_baseline_comparison(predictions_df)
        
        report = {
            'summary': {
                'total_predictions': len(predictions_df),
                'evaluation_period': f"{predictions_df.iloc[0]['period']} - {predictions_df.iloc[-1]['period']}"
            },
            'hit_rate_at_k': hit_rates,
            'expected_value': expected_value,
            'position_accuracy': position_accuracy,
            'baseline_comparison': baseline_comparison,
            'overall_accuracy': {
                'front_avg_hits': predictions_df['front_hits'].mean(),
                'back_avg_hits': predictions_df['back_hits'].mean(),
                'perfect_predictions': len(predictions_df[(predictions_df['front_hits'] == 5) & 
                                                          (predictions_df['back_hits'] == 2)])
            }
        }
        
        return report
    
    def print_report(self, report):
        """打印格式化的评估报告"""
        print("\n" + "="*60)
        print("📈 大乐透AI预测系统 - 评估报告")
        print("="*60)
        
        print(f"\n📅 评估概览:")
        print(f"  预测总数: {report['summary']['total_predictions']}")
        print(f"  评估期间: {report['summary']['evaluation_period']}")
        
        print(f"\n🎯 Hit-rate@k (Top-k候选命中率):")
        for key, value in report['hit_rate_at_k'].items():
            print(f"  {key}: {value:.2%}")
        
        print(f"\n💰 期望收益分析:")
        ev = report['expected_value']
        print(f"  总投资: ¥{ev['total_investment']:.2f}")
        print(f"  总奖金: ¥{ev['total_prize']:.2f}")
        print(f"  净收益: ¥{ev['net_profit']:.2f}")
        print(f"  ROI: {ev['roi_percent']:.2f}%")
        print(f"  单注期望收益: ¥{ev['expected_value_per_bet']:.2f}")
        
        print(f"\n📍 位置准确率:")
        print(f"  前区: {report['position_accuracy']['front']}")
        print(f"  后区: {report['position_accuracy']['back']}")
        
        print(f"\n🆚 AI vs 随机基准:")
        bc = report['baseline_comparison']
        print(f"  AI前区平均命中: {bc['ai_performance']['front_hits_avg']:.2f}")
        print(f"  随机前区平均命中: {bc['random_baseline']['front_hits_avg']:.2f}")
        print(f"  前区提升: {bc['improvement_percent']['front_improvement']:.2f}%")
        print(f"  AI后区平均命中: {bc['ai_performance']['back_hits_avg']:.2f}")
        print(f"  随机后区平均命中: {bc['random_baseline']['back_hits_avg']:.2f}")
        print(f"  后区提升: {bc['improvement_percent']['back_improvement']:.2f}%")
        
        print(f"\n📊 整体性能:")
        overall = report['overall_accuracy']
        print(f"  前区平均命中数: {overall['front_avg_hits']:.2f}")
        print(f"  后区平均命中数: {overall['back_avg_hits']:.2f}")
        print(f"  完美预测次数: {overall['perfect_predictions']}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    from database import Database
    from feature_engineering import FeatureEngineer
    from ml_models import LotteryPredictor
    
    print("🚀 启动MLOps评估系统...")
    
    db = Database()
    df = db.get_all_records()
    print(f"✅ 加载 {len(df)} 条历史数据")
    
    fe = FeatureEngineer(df)
    df_features = fe.extract_all_features()
    
    def predictor_func(train_df):
        predictor = LotteryPredictor()
        predictor.train_random_forest(train_df)
        predictor.train_xgboost(train_df)
        
        latest_features = train_df[['odd_count', 'even_count', 'large_count', 'small_count', 
                                    'front_sum', 'ac_value', 'span']].iloc[-1:].values
        
        result = predictor.predict_with_probabilities(latest_features, play_type='basic')
        pred = result['predictions'][0]
        
        return {
            'front': pred['front'],
            'back': pred['back']
        }
    
    evaluator = LotteryEvaluator()
    
    results = evaluator.walk_forward_validation(
        df_features, 
        predictor_func, 
        train_size=100, 
        test_size=20, 
        step=20
    )
    
    predictions_df = pd.DataFrame(results)
    
    report = evaluator.generate_evaluation_report(predictions_df)
    evaluator.print_report(report)
    
    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估报告已保存到 evaluation_report.json")
