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