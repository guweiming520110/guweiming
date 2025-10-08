# 多彩票AI预测系统

## Overview
This project is an AI-powered system designed to analyze and predict numbers for multiple Chinese lottery types, starting with "大乐透" (Super Lotto) and planning to expand to "双色球" (Double Color Ball) and "福彩3D" (3-digit lottery). It uses machine learning algorithms like Random Forest and XGBoost to process historical lottery data, extract features, and generate predictive number combinations. The system's purpose is data analysis and research, acknowledging the inherent randomness of lottery draws. It offers various prediction modes, including basic recommendations, multiple sets, high-probability selections, and "胆拖" (dare-drag) betting strategies, through a web interface for data collection, AI prediction, statistical analysis, and historical record viewing. The business vision is to provide a sophisticated tool for lottery data analysis and predictive modeling.

## User Preferences
- 沟通方式：简单、日常用语

## System Architecture
The system is built with Python 3.11 and Flask, using SQLAlchemy with a PostgreSQL database. Data processing leverages pandas and numpy, while web scraping uses Beautiful Soup 4 and requests for multiple sources with deduplication and fault tolerance.

**UI/UX Decisions:**
The web interface includes:
- **Data Collection Button:** One-click to fetch latest lottery data.
- **AI Prediction Functionality:** Generate recommended numbers based on chosen play styles.
- **Statistical Analysis:** Displays hot and cold numbers.
- **Historical Records:** Browsing of past lottery results.
- **Prediction Analysis UI:** View predictions vs. actual outcomes and hit rates.
- **Lottery Type Selector:** Allows selection between different lottery types (currently DLT, with SSQ and FC3D planned).

**Technical Implementations:**
- **Feature Engineering:** Utilizes 39 distinct features, including basic statistics, advanced patterns, temporal seasonality, and volatility indicators.
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
- **Persistence:** Models are persisted using `pickle`.
- **Database Schema:** `PredictionRecord` table includes fields for detailed "胆拖" (dare-drag) betting strategies and `lottery_type` for multi-lottery support.
- **Prediction Uniqueness:** Ensures only one prediction per target period and play type is stored.

## External Dependencies
- **Backend Framework:** Flask
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Data Analysis & Manipulation:** pandas, numpy
- **Web Scraping:** BeautifulSoup4, requests
- **Machine Learning:** scikit-learn, XGBoost, pickle
- **Deep Learning (Optional):** TensorFlow (for LSTM and Transformer models)
- **Visualization:** Plotly