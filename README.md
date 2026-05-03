# RUL Predictive Maintenance System (Kataoka Robotics)

## Overview
This project builds an end-to-end machine learning system to estimate Remaining Useful Life (RUL) of industrial robotic components using time-series sensor data.

## Key Features
- Time-series alignment of sensor, maintenance, and failure data
- Advanced feature engineering (degradation trends, stress indices)
- ML models (Random Forest, Gradient Boosting, Extra Trees)
- MLflow experiment tracking
- Achieved R² = 0.64 on test data

## Model Performance
- Best Model: Gradient Boosting
- Test R²: 0.64
- MAE: ~64 hours

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn
- MLflow
- Gradio (deployment)

## Pipeline
1. Data ingestion
2. Time-series alignment
3. Feature engineering
4. Model training & tracking
5. Deployment (Gradio)

## Note
Large datasets and models are excluded from GitHub for performance reasons.

## Latest Update
- Added real-time risk scoring engine
- Added risk breakdown visualization
- Improved health classification logic
- Added explainability layer

## 🔗 Author
Uzor Nwokeaka
