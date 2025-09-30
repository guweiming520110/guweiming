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
        df = db.get_all_records()
        if df.empty:
            return jsonify({'error': '暂无数据，无法预测'}), 404
        
        if len(df) < 50:
            return jsonify({'error': f'数据量不足：需要至少50条数据才能进行预测，当前只有{len(df)}条'}), 400
        
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
        
        prediction = predictor.predict_numbers(latest_features)
        
        return jsonify({
            'prediction': prediction,
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
