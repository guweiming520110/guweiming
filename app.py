from flask import Flask, render_template, jsonify, request
import os
from database import Database
from feature_engineering import FeatureEngineer
from ml_models import LotteryPredictor
import pandas as pd
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)
db = Database()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/historical_data')
def get_historical_data():
    df = db.get_all_records()
    if df.empty:
        return jsonify({'error': '暂无历史数据，请先采集数据'}), 404
    
    limit = request.args.get('limit', default=100, type=int)
    total_count = len(df)
    
    records = df.head(limit).to_dict('records')
    
    return jsonify({
        'data': records,
        'total_count': total_count
    })

@app.route('/api/statistics')
def get_statistics():
    df = db.get_all_records()
    if df.empty:
        return jsonify({'error': '暂无数据'}), 404
    
    fe = FeatureEngineer(df)
    fe.add_number_frequency()
    hot_cold = fe.get_hot_cold_numbers()
    
    return jsonify({
        'hot_front': hot_cold['hot_front'],
        'cold_front': hot_cold['cold_front'],
        'hot_back': hot_cold['hot_back'],
        'cold_back': hot_cold['cold_back']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        from datetime import datetime
        
        df = db.get_all_records()
        if df.empty:
            return jsonify({'error': '暂无数据，无法预测'}), 404
        
        if len(df) < 50:
            return jsonify({'error': f'数据量不足：需要至少50条数据才能进行预测，当前只有{len(df)}条'}), 400
        
        data = request.get_json() or {}
        play_type = data.get('play_type', 'basic')
        
        fe = FeatureEngineer(df)
        df_features = fe.extract_all_features()
        
        predictor = LotteryPredictor()
        
        model_exists = os.path.exists('models/rf_front.pkl')
        
        if model_exists:
            try:
                predictor.load_models()
                predictor.validate_models(df_features)
                print("使用已有模型进行预测")
            except Exception as e:
                print(f"模型验证失败: {e}")
                print("重新训练模型...")
                predictor = LotteryPredictor()
                predictor.train_random_forest(df_features)
                predictor.train_xgboost(df_features)
                predictor.save_models()
        else:
            print("首次训练模型...")
            predictor.train_random_forest(df_features)
            predictor.train_xgboost(df_features)
            predictor.save_models()
        
        feature_cols = ['odd_count', 'even_count', 'large_count', 'small_count', 
                       'front_sum', 'ac_value', 'span', 'max_front_omission', 
                       'avg_front_omission', 'max_back_omission', 'avg_back_omission']
        
        available_cols = [col for col in feature_cols if col in df_features.columns]
        if not available_cols:
            available_cols = ['front_1', 'front_2', 'front_3', 'front_4', 'front_5']
        
        latest_features = df_features[available_cols].iloc[-1:].values
        
        prediction_result = predictor.predict_with_probabilities(latest_features, play_type=play_type)
        
        latest_period_str = db.get_latest_period()
        if latest_period_str:
            target_period = str(int(latest_period_str) + 1)
        else:
            target_period = "00001"
        predict_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for idx, pred in enumerate(prediction_result['predictions']):
            prediction_data = {
                'predict_time': predict_time,
                'target_period': target_period,
                'play_type': play_type,
                'front_numbers': ','.join(map(str, pred['front'])),
                'back_numbers': ','.join(map(str, pred['back'])),
                'front_probabilities': json.dumps(pred['front_probs']),
                'back_probabilities': json.dumps(pred['back_probs'])
            }
            db.save_prediction(prediction_data)
        
        return jsonify({
            'prediction': prediction_result,
            'target_period': target_period,
            'message': '预测号码仅供参考，彩票开奖是随机的',
            'data_count': len(df)
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"预测错误: {error_detail}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/frequency')
def frequency_chart():
    df = db.get_all_records()
    if df.empty:
        return jsonify({'error': '暂无数据'}), 404
    
    fe = FeatureEngineer(df)
    fe.add_number_frequency()
    
    front_nums = list(range(1, 36))
    front_freqs = [fe.front_freq[num] for num in front_nums]
    
    fig = go.Figure(data=[
        go.Bar(x=front_nums, y=front_freqs, name='前区号码频率')
    ])
    
    fig.update_layout(
        title='前区号码出现频率统计',
        xaxis_title='号码',
        yaxis_title='出现次数',
        template='plotly_white'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'chart': graphJSON})

@app.route('/api/predictions/history')
def get_predictions_history():
    try:
        limit = request.args.get('limit', default=50, type=int)
        predictions = db.get_predictions(limit=limit)
        
        result = []
        for pred in predictions:
            result.append({
                'id': pred.id,
                'predict_time': pred.predict_time,
                'target_period': pred.target_period,
                'play_type': pred.play_type,
                'front_numbers': pred.front_numbers,
                'back_numbers': pred.back_numbers,
                'front_probabilities': pred.front_probabilities,
                'back_probabilities': pred.back_probabilities,
                'actual_front': pred.actual_front,
                'actual_back': pred.actual_back,
                'is_matched': pred.is_matched,
                'match_level': pred.match_level,
                'front_match_count': pred.front_match_count,
                'back_match_count': pred.back_match_count
            })
        
        return jsonify({'predictions': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/verify', methods=['POST'])
def verify_predictions():
    try:
        df = db.get_all_records()
        if df.empty:
            return jsonify({'error': '暂无历史开奖数据'}), 404
        
        predictions = db.get_predictions(limit=1000)
        verified_count = 0
        
        for pred in predictions:
            if pred.actual_front:
                continue
            
            target_period = pred.target_period
            actual_record = df[df['period'] == target_period]
            
            if not actual_record.empty:
                row = actual_record.iloc[0]
                actual_front = [row['front_1'], row['front_2'], row['front_3'], row['front_4'], row['front_5']]
                actual_back = [row['back_1'], row['back_2']]
                
                pred_front = list(map(int, pred.front_numbers.split(',')))
                pred_back = list(map(int, pred.back_numbers.split(',')))
                
                front_match = len(set(pred_front) & set(actual_front))
                back_match = len(set(pred_back) & set(actual_back))
                
                actual_front_str = ','.join(map(str, sorted(actual_front)))
                actual_back_str = ','.join(map(str, sorted(actual_back)))
                
                db.update_prediction_result(
                    pred.id,
                    actual_front_str,
                    actual_back_str,
                    front_match,
                    back_match
                )
                verified_count += 1
        
        return jsonify({
            'success': True,
            'verified_count': verified_count,
            'message': f'已验证 {verified_count} 条预测记录'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scrape', methods=['POST'])
def scrape_data():
    try:
        from multi_source_scraper import MultiSourceScraper
        
        data = request.get_json() or {}
        limit = data.get('limit', 5000)
        
        old_count = db.get_record_count()
        
        # 使用多数据源采集器
        scraper = MultiSourceScraper()
        result = scraper.scrape_all_sources(limit=limit)
        
        df = result['dataframe']
        sources = result['sources']
        
        if not df.empty:
            # 保存到CSV
            filename = 'lottery_data.csv'
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            # 导入数据库
            db.insert_data_from_csv(filename)
            
            new_count = db.get_record_count()
            added_count = new_count - old_count
            
            period_range = f"{df['period'].iloc[0]} 到 {df['period'].iloc[-1]}"
            
            return jsonify({
                'success': True,
                'message': f'成功采集 {len(df)} 条数据，新增 {added_count} 条记录',
                'scraped_count': len(df),
                'added_count': added_count,
                'total_count': new_count,
                'period_range': period_range,
                'latest_period': db.get_latest_period(),
                'sources': sources
            })
        else:
            return jsonify({'error': '数据采集失败'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
