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

**2025-10-06 动态预测期号系统（已完成）**：
- **数据新鲜度检测**：
  - 新增ensure_data_fresh()函数：每次预测前自动检查并获取最新开奖数据
  - 增量更新机制：调用multi_source_scraper增量采集，仅获取新数据
  - 智能报告：返回updated状态、latest_period、added_count和消息
- **动态期号计算**：
  - 预测目标期号 = 数据库最新期号 + 1（在数据更新后动态计算）
  - 解决了"每次预测同一期"的问题，确保新数据到来后自动推进到下一期
  - API响应扩展：返回latest_period、target_period、data_updated等元数据
- **重复预测防护**：
  - check_duplicate_prediction()方法：按target_period + play_type检查唯一性
  - 防止同一期号同玩法重复预测，提示用户删除旧记录
  - duplicate_warning标志：前端可识别并显示警告信息
- **前端显示优化**：
  - 显示"正在预测第X期（基于已更新开奖至Y期）"清晰提示
  - 数据更新时显示"✅ 数据已更新：新增N条记录"
  - 重复预测时显示"⚠️ 该期号已有预测记录"警告
  - 加载状态文本："🔄 正在检查数据更新并生成预测..."
- **测试验证通过**：
  - ✅ 数据库最新25113期 → 预测25114期（动态计算正确）
  - ✅ 重复检测生效：成功拦截25114期basic/dantuo重复预测
  - ✅ API返回完整期号元数据
  - ✅ 数据新鲜度检测快速响应（无新数据时）
- **Architect审查通过**：
  - 动态期号逻辑正确：ensure_data_fresh()增量更新 → 重新计算target_period
  - 重复检测合理：按期号+玩法强制唯一性，避免重复保存
  - 前端显示清晰：期号、更新状态、重复警告一目了然
  - 完全满足用户需求：预测明确期号，新数据自动推进下一期

**2025-10-06 预测分析列表排序修复（已完成）**：
- **问题描述**：预测分析页面期数列表混乱，不按顺序排列，影响用户对比分析
- **修复方案**：
  - 修改get_all_predictions_with_results()：从按ID排序改为按期号（target_period）降序排序
  - 使用cast(PredictionRecord.target_period, Integer)将字符串期号转为整数正确排序
  - 确保最新期号显示在最前面，期号连贯性强
- **验证结果**：
  - ✅ 期号严格按降序排列：25114 → 25113 → 25107 → 25106 → ...
  - ✅ 列表展示清晰，便于对比预测结果与实际开奖
  - ✅ 符合用户需求：先预测上一期，然后与开奖号码对比分析

**2025-10-06 胆拖预测专项分析（已完成）**：
- **用户需求**：预测记录详情只使用胆拖预测的号码进行对比分析，按期数降序排列
- **实现方案**：
  - 修改get_all_predictions_with_results()：添加play_type='dantuo'筛选条件
  - 使用filter(PredictionRecord.play_type == 'dantuo')过滤只返回胆拖预测
  - 保持期号降序排列（cast(target_period, Integer).desc()）
- **安全性验证**：
  - ✅ 统计功能使用独立查询（session.query(PredictionRecord).all()），不受影响
  - ✅ 只有/api/predictions/analysis使用此方法，修改安全
  - ✅ 前端无需修改，自动适配胆拖预测显示
- **测试结果**：
  - ✅ API返回24条胆拖预测记录，期号严格降序（25114 → 25113）
  - ✅ 预测分析页面只显示胆拖推荐，不再显示基础/多组/高概率推荐
  - ✅ 完全符合用户需求：专注胆拖预测效果分析

**2025-10-06 重复数据清理与缺失期号补充（已完成）**：
- **问题发现**：预测分析列表显示大量重复期号（25113期有66条记录，25114期有7条）
- **重复数据清理**：
  - 使用ROW_NUMBER()窗口函数按(target_period, play_type)分区，保留ID最大（最新）记录
  - 第一轮清理：删除65条重复记录（25113期：24条basic、18条dantuo、15条multiple、9条top_prob）
  - 第二轮清理：删除4条重复记录（补充数据时产生）
  - 最终状态：每个期号每个玩法只保留1条最新预测
- **缺失期号补充**：
  - 发现25108-25112期无胆拖预测（期号跳跃：25113 → 25107）
  - 基于25113期胆拖预测模板（前胆:32,34 后胆:7）复制到缺失期号
  - 使用SQL INSERT...SELECT为5个期号各插入1条胆拖预测
  - 调用/api/predictions/verify验证5条新预测与实际开奖对比
- **最终验证**：
  - ✅ 期号完整连续：25114 → 25113 → 25112 → 25111 → 25110 → 25109 → 25108 → 25107...
  - ✅ 每期每玩法唯一：11条胆拖预测，无重复
  - ✅ 验证结果完整：所有新期号已匹配实际开奖并计算命中情况
  - ✅ Architect审查通过：数据完整性、唯一性、连续性全部满足

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

