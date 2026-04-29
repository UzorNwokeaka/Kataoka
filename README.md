# RUL Predictive Maintenance System

End-to-end machine learning pipeline for predicting Remaining Useful Life (RUL) of industrial robotic systems using time-series sensor data.

## Features
- Data integration across multiple tables
- Synthetic data augmentation for sparse failure events
- RUL target engineering
- Feature engineering (rolling statistics, degradation trends)
- ML-ready dataset generation

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- MLflow (planned)
- Gradio (planned)

## Project Structure

## Data Preprocessing

- Handled missing values using forward/backward fill per robot
- Removed unrealistic sensor values
- Applied IQR-based outlier capping
- Standardized numerical features using StandardScaler
- Encoded categorical variables using one-hot encoding
- Produced final ML-ready dataset

## Dataset

- Full dataset is generated via pipeline:
  - `build_rul_dataset.py`
  - `preprocess_data.py`

- A sample dataset is provided for quick testing:
  - `data/sample/sample_data.csv`