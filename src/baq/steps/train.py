"""
Training Module for Bangkok Air Quality Forecasting.

This module provides functionality to train various models for PM2.5 forecasting.
It supports both tabular models (XGBoost, Random Forest) and LSTM models.

Features:
- Multi-model training support (LSTM, Random Forest, XGBoost)
- Automatic data preparation for different model types
- Sequence creation for LSTM models with sliding window approach
- Tabular feature engineering for traditional ML models
- Model validation and performance evaluation
- Configurable training parameters and hyperparameters
- Comprehensive logging and error handling
- Model-specific optimization strategies

The module handles the complete training pipeline:
1. Data preparation based on model type (tabular vs sequential)
2. Model instantiation with custom parameters
3. Training execution with validation monitoring
4. Performance evaluation on test data
5. Model artifact preparation for saving

Example:
    >>> model, metrics = train_model(
    ...     X_train=train_features,
    ...     y_train=train_targets,
    ...     X_val=val_features,
    ...     y_val=val_targets,
    ...     X_test=test_features,
    ...     y_test=test_targets,
    ...     model_name="lstm",
    ...     model_params=lstm_config,
    ...     model_training_params=training_config,
    ...     training_config=config
    ... )
"""


import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Union, List
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from baq.core.evaluation import calculate_metrics
from baq.data.utils import create_sequences
from baq.models.lstm import LSTMForecaster

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    model_params: dict,
    model_training_params: dict,
    training_config: dict
) -> Tuple[object, dict]:
    """
    Train a model based on the specified model name and parameters.
        if ML model -> train on tabular lag-features
        if LSTM     -> sliding window + LSTM

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model to train (e.g., "xgboost", "random_forest", "lstm")
        model_params: Parameters for the model
        model_training_params: Configuration for training (e.g., epochs, batch size)
        training_config: Configuration for training 

    Returns:
        model: Trained model
        metrics: Evaluation metrics
    """
    model_name = model_name.lower()
    if model_name in ("xgboost", "random_forest"):
        if model_name == "xgboost":
            model = XGBRegressor(**model_params)
        else:
            model = RandomForestRegressor(**model_params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calculate_metrics(y_test, preds)
        return model, metrics

    elif model_name == "lstm":
        seq_len = int(training_config.get("sequence_length", 24))
        # 1) create sequence
        X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
        X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len)

        # 2) build & train LSTM
        lstm_params = {
            "input_shape": (seq_len, X_train.shape[1]),
            "lstm_units": model_params.get("lstm_units", (128, 64)),
            "dropout_rate": model_params.get("dropout_rate", 0.2),
            "learning_rate": model_params.get("learning_rate", 1e-3),
            "checkpoint_path": model_params.get("checkpoint_path", "best_lstm.h5"),
            "early_stopping_patience": int(model_training_params.get("early_stopping_patience", 10)),
            "reduce_lr_patience": int(model_training_params.get("reduce_lr_patience", 5))
        }
        
        model = LSTMForecaster(**lstm_params)
        model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=int(model_training_params.get("epochs", 50)),
            batch_size=int(model_training_params.get("batch_size", 32)),
            shuffle=False,
            verbose=1
        )

        # 3) evaluate
        preds = model.predict(X_te_seq)
        metrics = calculate_metrics(y_te_seq, preds)
        return model, metrics

    else:
        raise ValueError(f"Unsupported model: {model_name}")
