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
    
    records = df.head(50).to_dict('records')
    return jsonify({'data': records})

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
        df = db.get_all_records()
        if df.empty:
            return jsonify({'error': '暂无数据，无法预测'}), 404
        
        fe = FeatureEngineer(df)
        df_features = fe.extract_all_features()
        
        predictor = LotteryPredictor()
        
        if os.path.exists('models/rf_front.pkl'):
            predictor.load_models()
        else:
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
        
        prediction = predictor.predict_numbers(latest_features)
        
        return jsonify({
            'prediction': prediction,
            'message': '预测号码仅供参考，彩票开奖是随机的'
        })
    
    except Exception as e:
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

@app.route('/api/scrape', methods=['POST'])
def scrape_data():
    try:
        from data_scraper import DaLeTouScraper
        
        scraper = DaLeTouScraper()
        df = scraper.scrape_historical_data(start_year=2023, end_year=2025)
        
        if not df.empty:
            filename = scraper.save_to_csv(df)
            db.insert_data_from_csv(filename)
            return jsonify({
                'success': True,
                'message': f'成功采集 {len(df)} 条数据（2023-2025年）',
                'count': len(df)
            })
        else:
            return jsonify({'error': '数据采集失败'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
