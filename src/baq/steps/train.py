"""
This module contains the code for training forecasting models.
- Implements various time series forecasting algorithms
- Handles model hyperparameter tuning and optimization
- Supports both traditional statistical models and machine learning approaches
- Manages the training process including data splitting, validation, and model persistence
- Provides utilities for handling time-based cross-validation and seasonal patterns
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from baq.core.evaluation import calculate_metrics


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_params: dict,
    training_config: dict,
) -> tuple[object, dict]:
    """
    Train the model.
    """
    # 1. Time Series Split for evaluation only
    # Use TimeSeriesSplit for proper time series validation
    # This ensures we're always training on past data and testing on future data
    # Convert test_size from percentage to number of samples if it's a float
    test_size = training_config["test_size"]
    if isinstance(test_size, float) and 0 < test_size < 1:
        test_size = int(len(X) * test_size)
    
    tscv = TimeSeriesSplit(n_splits=training_config["n_splits"], test_size=test_size)
    
    # Dictionary to store all results
    all_metrics = {}
    all_metrics[model_name] = {}
    
    # Evaluate model performance using cross-validation
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nEvaluating on fold {fold+1}/{training_config['n_splits']}")
        
        # Split data for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train model for evaluation
        fold_model = None
        if model_name == "xgboost":
            fold_model = XGBRegressor(**model_params).fit(X_train, y_train)
        elif model_name == "random_forest":
            fold_model = RandomForestRegressor(**model_params).fit(X_train, y_train)
        
        # Evaluate model for this fold
        print(f"\nEvaluating {model_name} model (fold {fold+1}):")
        y_pred = fold_model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        all_metrics[model_name][fold] = metrics

    # Calculate average metrics across all folds
    avg_metrics = {model_name: {}}
    for metric in list(all_metrics[model_name][0].keys()):
        avg_metrics[model_name][metric] = sum(fold_data[metric] for fold_data in all_metrics[model_name].values()) / len(all_metrics[model_name])
    
    # Now train the final model on the full dataset
    print(f"\nTraining final {model_name} model on full dataset")
    final_model = None
    if model_name == "xgboost":
        final_model = XGBRegressor(**model_params).fit(X, y)
    elif model_name == "random_forest":
        final_model = RandomForestRegressor(**model_params).fit(X, y)

    return final_model, avg_metrics