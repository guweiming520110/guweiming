# 多彩票AI预测系统

## Overview
This project is an AI-powered system designed to analyze and predict numbers for multiple Chinese lottery types, starting with "大乐透" (Super Lotto) and planning to expand to "双色球" (Double Color Ball) and "福彩3D" (3-digit lottery). It uses machine learning algorithms like Random Forest and XGBoost to process historical lottery data, extract features, and generate predictive number combinations. The system's purpose is data analysis and research, acknowledging the inherent randomness of lottery draws. It offers various prediction modes, including basic recommendations, multiple sets, high-probability selections, and "胆拖" (dare-drag) betting strategies, through a web interface for data collection, AI prediction, statistical analysis, and historical record viewing. The business vision is to provide a sophisticated tool for lottery data analysis and predictive modeling.

## User Preferences
- 沟通方式：简单、日常用语

## System Architecture
The system is built with Python 3.11 and Flask, using SQLAlchemy with a PostgreSQL database. Data processing leverages pandas and numpy, while web scraping uses Beautiful Soup 4 and requests for multiple sources with deduplication and fault tolerance.

**Multi-Lottery Architecture Status (Plan A - Progressive Implementation):**
- ✅ **大乐透 (DLT):** Fully operational with complete data pipeline, AI prediction, and evaluation
- 🔧 **双色球 (SSQ):** Infrastructure ready, requires scraper implementation and feature engineering refinements
- 🔧 **福彩3D (FC3D):** Infrastructure ready, requires scraper implementation and feature engineering adaptations for repeated digits

**UI/UX Decisions:**
The web interface includes:
- **Data Collection Button:** One-click to fetch latest lottery data.
- **AI Prediction Functionality:** Generate recommended numbers based on chosen play styles.
  - **基础推荐 (Basic):** 1 set of 5 front + 2 back numbers
  - **多组推荐 (Multiple):** 3 sets of combinations
  - **复式推荐 (Top Probability):** 10 front + 5 back high-probability numbers
  - **胆拖推荐 (Dare-Drag):** 2胆6拖 for front zone + 1胆3拖 for back zone (60 combinations)
- **Statistical Analysis:** Displays hot and cold numbers.
- **Historical Records:** Browsing of past lottery results.
- **Prediction Analysis UI:** View predictions vs. actual outcomes and hit rates.
- **Lottery Type Selector:** Allows selection between different lottery types (currently DLT, with SSQ and FC3D planned).

**Technical Implementations:**
- **Feature Engineering:** Utilizes 39 distinct features, including basic statistics, advanced patterns, temporal seasonality, and volatility indicators. Fully parameterized to support different lottery types with dynamic column generation and range-aware calculations. Note: Some advanced features (span, consecutive, interval) currently optimized for non-repeating number sequences (大乐透, 双色球), may need refinement for 福彩3D with repeated digits.
- **AI Prediction Models:**
    - **Ensemble Approach:** Weighted integration of Random Forest (40%) and XGBoost (60%).
    - **Deep Learning (Optional):** Future integration of Bi-directional LSTM and Transformer models for a four-model ensemble.
    - **Model Training:** Independent models for each number position, trained on historical data.
    - **Prediction Logic:** Combines outputs, prioritizing highest frequency candidates.
    - **Model Validation:** Automated detection of model-data mismatches and retraining.
    - **Feature Standardization:** StandardScaler for normalization.
- **Data Workflow:** Crawler collection → Data cleaning → Database storage → Feature extraction → Model training → Prediction generation.
- **MLOps Evaluation System:** Features a walk-forward time-series validation framework, hit-rate@k metrics, expected value calculation, and per-position accuracy.
- **Dynamic Period Calculation:** Automatically determines the target prediction period by checking the latest data and incrementing.
- **Prediction Validation System:** Compares generated predictions against actual lottery results, calculating hit rates and potential prize tiers.

**System Design Choices:**
- **Modularity:** Clear separation of concerns for scraping, database, feature engineering, and ML models.
- **Scalability:** Designed to handle large data volumes and multiple sources.
- **Persistence:** Models are persisted using `pickle` in lottery-type-specific subdirectories (`models/dlt/`, `models/ssq/`, `models/fc3d/`).
- **Database Schema:** `PredictionRecord` table includes fields for detailed "胆拖" (dare-drag) betting strategies and `lottery_type` for multi-lottery support. Composite indexes on `(lottery_type, period)` and `(lottery_type, target_period)` ensure data integrity.
- **Prediction Uniqueness:** Ensures only one prediction per target period, play type, and lottery type is stored.
- **Dynamic Configuration:** All core modules (MultiSourceScraper, FeatureEngineer, LotteryPredictor) accept `lottery_type` parameter and adapt behavior based on `lottery_config.py` rules.
- **Parameterization:** Eliminates hardcoded assumptions about front/back ball counts, number ranges, and thresholds. Fallback logic dynamically generates feature columns based on current lottery configuration.

## External Dependencies
- **Backend Framework:** Flask
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Data Analysis & Manipulation:** pandas, numpy
- **Web Scraping:** BeautifulSoup4, requests
- **Machine Learning:** scikit-learn, XGBoost, pickle
- **Deep Learning (Optional):** TensorFlow (for LSTM and Transformer models)
- **Visualization:** Plotly