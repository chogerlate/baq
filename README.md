# PM2.5 Forecasting Pipeline

This project implements a machine learning pipeline for forecasting PM2.5 air quality levels using time series data, tracked with Weights & Biases and managed with Hydra configuration.

## Features

- **Time Series Forecasting**: Predict PM2.5 levels hours in advance
- **Feature Engineering**: Automatic creation of time-based features, lag features, and rolling statistics
- **Model Training**: Support for XGBoost, Random Forest, and other regression models
- **Evaluation**: Comprehensive metrics and visualizations for model performance
- **Experiment Tracking**: Full integration with Weights & Biases for experiment tracking
- **Configuration Management**: Hydra for flexible configuration of all pipeline components

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pm25-forecasting.git
cd pm25-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## Usage

### Basic Training

To train a model with default settings:

```bash
python src/train.py
```

### Using Different Models

To train with a specific model:

```bash
python src/train.py model=random_forest
```

### Modifying Parameters

To change the forecast horizon:

```bash
python src/train.py data.forecast_horizon=12
```

### Configuration

The pipeline uses Hydra for configuration management. Main configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/xgboost.yaml`: XGBoost model configuration
- `configs/model/random_forest.yaml`: Random Forest model configuration

## Project Structure

```
pm25-forecasting/
├── configs/              # Configuration files
│   ├── config.yaml       # Main configuration
│   └── model/            # Model-specific configurations
├── src/                  # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── utils/            # Utility functions
│   └── train.py          # Main training script
├── notebooks/            # Jupyter notebooks for exploration
├── artifacts/            # Output artifacts (models, plots, etc.)
└── requirements.txt      # Dependencies
```

## Outputs

After training, the pipeline produces the following artifacts:

- Trained model and scaler
- Performance metrics
- Visualizations (feature importance, actual vs. predicted plots)
- Model card with key information
- PM2.5 forecasts for future periods
