# Project ARIES

A stock price prediction and trading experiment. The project trains neural-network models (Keras/TensorFlow LSTMs and BiLSTMs) and XGBoost models to forecast stock prices and generate trading signals, applied across Dow Jones components and selected Malaysian stocks.

## Contents
- `Apple Trading v1/`, `Tesla Trading v1/`, `UEM Sunrise v1/` — single-stock training and trading pipelines with notebooks and saved models.
- `Parallel Trading v2/` — a multi-stock pipeline with per-stock models and scalers.
- `beta/` — a backtesting experiment with per-stock models at multiple complexity levels, scalers, and trade logs.

## Tech
Python, Jupyter, Keras/TensorFlow, XGBoost, scikit-learn.

## Note
Model and scaler files are saved artifacts from local experiments. Results are experimental and are not investment advice.
