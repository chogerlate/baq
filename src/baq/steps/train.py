import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Union, List
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from baq.core.evaluation import calculate_metrics
from baq.data.utils import create_sequences
from baq.models.lstm import create_lstm_model, create_lstm_callbacks
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
        training_config: Configuration for training (e.g., epochs, batch size)
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
        X_val_seq, y_val_seq = create_sequences(X_val,   y_val,   seq_len)
        X_te_seq,  y_te_seq  = create_sequences(X_test,  y_test,  seq_len)

        # 2) build & train LSTM
        model = create_lstm_model(input_shape=(seq_len, X_train.shape[1]))
        callbacks = create_lstm_callbacks(
            early_stopping_patience=int(training_config.get("early_stopping_patience", 10)),
            reduce_lr_patience=int(training_config.get("reduce_lr_patience", 5))
        )
        model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=int(training_config.get("epochs", 50)),
            batch_size=int(training_config.get("batch_size", 32)),
            callbacks=callbacks,
            shuffle=False,
            verbose=1
        )

        # 3) evaluate
        preds = model.predict(X_te_seq).reshape(-1)
        metrics = calculate_metrics(y_te_seq, preds)
        return model, metrics

    else:
        raise ValueError(f"Unsupported model: {model_name}")
