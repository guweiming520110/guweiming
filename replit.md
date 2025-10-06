# 大乐透AI预测系统

## Overview
This project is an AI-powered system designed to analyze and predict numbers for China's "大乐透" (Super Lotto) lottery. Utilizing machine learning algorithms like Random Forest and XGBoost, the system processes historical lottery data, extracts complex features, and generates predictive number combinations. The primary purpose is data analysis and research, acknowledging the inherent randomness of lottery draws. The system provides various prediction modes, including basic recommendations, multiple sets, high-probability selections, and "胆拖" (dare-drag) betting strategies. It features a web interface for data collection, AI prediction, statistical analysis, and viewing historical records.

## User Preferences
- 沟通方式：简单、日常用语

## System Architecture
The system employs a robust backend built with Python 3.11 and Flask, using SQLAlchemy for ORM with a PostgreSQL database. Data processing relies on pandas and numpy, while Beautiful Soup 4 and requests handle web scraping from multiple sources (500彩票网, 新浪彩票, 体彩官网) with intelligent deduplication and fault tolerance.

**UI/UX Decisions:**
The web interface provides clear, interactive elements for user interaction:
- **Data Collection Button:** One-click functionality to fetch the latest lottery data.
- **AI Prediction Functionality:** Allows users to generate recommended numbers based on chosen play styles (Basic 5+2, Multiple sets, High-probability Top N, Dare-drag).
- **Statistical Analysis:** Displays hot and cold numbers.
- **Historical Records:** Browsing of past lottery results.
- **Prediction Analysis UI**: New section to view predictions vs actual outcomes and visualize statistics like hit rates.

**Technical Implementations:**
- **Feature Engineering:** An advanced system with 39 distinct features, including basic statistics (frequency, omissions, odd/even, big/small, AC value, sum, span), advanced patterns (co-occurrence, temporal seasonality, consecutive numbers, interval density, volatility indicators, omission extension), short-term frequency momentum, conditional interval features, and period transition features.
- **AI Prediction Models:**
    - **Ensemble Approach:** Weighted integration of Random Forest (40%) and XGBoost (60%) models.
    - **Deep Learning (Optional):** Integration of Bi-directional LSTM and Transformer models (when available) for further enhancement, forming a four-model ensemble (RF 20%, XGBoost 20%, LSTM 30%, Transformer 30%).
    - **Model Training:** Independent models for each number position, trained on all historical data.
    - **Prediction Logic:** Combines outputs from multiple models, prioritizing highest frequency candidates.
    - **Model Validation:** Automated detection of model-data mismatches and retraining.
    - **Feature Standardization:** StandardScaler used for feature normalization.
- **Data Workflow:** Crawler collection → Data cleaning → Database storage → Feature extraction → Model training → Prediction generation.
- **MLOps Evaluation System:** Includes a walk-forward time-series validation framework, hit-rate@k metrics, expected value calculation, and per-position accuracy, with API endpoints for execution and reporting.

**System Design Choices:**
- **Modularity:** Clear separation of concerns with dedicated modules for scraping, database, feature engineering, and ML models.
- **Scalability:** Designed to handle large volumes of historical data and multiple data sources.
- **Persistence:** Models are persisted using `pickle`.
- **Database Schema:** `PredictionRecord` table enhanced with 5 "胆拖" fields for detailed betting strategy storage.

## External Dependencies
- **Backend Framework:** Flask
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Data Analysis & Manipulation:** pandas, numpy
- **Web Scraping:** BeautifulSoup4, requests
- **Machine Learning:** scikit-learn, XGBoost, pickle
- **Deep Learning (Optional):** TensorFlow (for LSTM and Transformer models)
- **Visualization:** Plotly
## Recent Changes

**2025-10-06 预测验证与分析系统（已完成）**：
- **数据库架构升级**：
  - PredictionRecord表新增5个胆拖字段：front_dan, front_tuo, back_dan, back_tuo, combination_count
  - 执行ALTER TABLE成功添加新列到生产数据库
  - 历史数据回填：成功解析并回填18条旧胆拖记录
- **预测保存修复**：
  - 修复save_prediction方法：现在正确接受并保存所有胆拖字段
  - 修复/api/predict接口：在dantuo模式下传递完整胆拖数据到数据库
  - 验证通过：新预测（ID 68, 2025-10-06）正确保存所有胆拖字段
- **预测验证系统**：
  - `/api/predictions/verify`：自动从数据库获取实际开奖数据并与预测对比
  - 智能区分胆拖和普通玩法，正确解析"胆:X 拖:Y"格式
  - 计算前区、后区命中数，自动判定中奖等级（一等奖～八等奖）
  - 测试验证：已成功验证66/68条历史预测
- **胆码全中检测**：
  - 使用set().issubset()算法检测胆码是否全部命中
  - 分别统计前区胆码全中率、后区胆码全中率、双区胆码全中率
  - 测试数据：17个胆拖预测，前区0%，后区5.9%，双区0%
- **预测分析API实现**：
  - `/api/predictions/analysis`：返回所有预测记录及其与实际开奖的对比结果
  - `/api/predictions/statistics`：计算各玩法命中率、胆码全中率等统计指标
  - 支持按玩法分类统计：basic, multiple, top_prob, dantuo
- **前端预测分析UI**：
  - 新增"验证预测结果"红色按钮：一键验证所有预测
  - "查看预测分析"按钮：显示预测vs实际对比表格
  - "查看统计数据"按钮：可视化命中率、胆拖成功率等图表
  - 显示预测历史记录：期号、玩法、预测号码、实际结果、命中情况、中奖等级
  - 验证成功后自动刷新分析和统计数据
- **Architect审查通过**：
  - 验证API正确对比预测与实际开奖
  - 胆码全中检测算法准确（issubset方法）
  - 前端验证流程完整，66/68条记录成功验证
  - 完全满足用户需求：预测vs实际对比分析，特别是胆拖命中效果

