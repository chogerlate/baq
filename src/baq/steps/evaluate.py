"""
Model Evaluation Module for Bangkok Air Quality Forecasting.

This module provides comprehensive evaluation capabilities for PM2.5 forecasting models.
It supports both single-step and multi-step forecasting evaluation with detailed
performance metrics and visualization.

Features:
- Single-step and multi-step forecasting evaluation
- Support for multiple model types (LSTM, Random Forest, XGBoost)
- Comprehensive performance metrics calculation
- Visualization of predictions vs actual values
- Sequence handling for time series models
- Model-agnostic evaluation interface

The evaluation process includes:
1. Single-step forecasting and metrics calculation
2. Multi-step forecasting and metrics calculation
3. Performance visualization generation
4. Comparative analysis between forecasting approaches

Example:
    >>> single_metrics, multi_metrics, plots = evaluate_model(
    ...     model=trained_model,
    ...     X_test=test_features,
    ...     y_test=test_targets,
    ...     forecast_horizon=24,
    ...     sequence_length=24
    ... )

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model as KerasModel
from typing import Dict, Tuple, Any

from baq.core.evaluation import calculate_metrics
from baq.core.inference import single_step_forecasting, multi_step_forecasting
from baq.models.lstm import LSTMForecaster


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    forecast_horizon: int,
    sequence_length: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, plt.Figure]]:
    """
    Evaluate the model on test data using both single-step and multi-step forecasting.

    Args:
        model: The trained model (LSTMForecaster, Keras Model, sklearn, or XGBoost)
        X_test: Test features DataFrame
        y_test: Test target Series
        forecast_horizon: Number of steps for multi-step forecasting
        sequence_length: Length of sequence for LSTM models

    Returns:
        Tuple containing:
        - single_step_metrics: Dict of metrics for single-step forecasting
        - multi_step_metrics: Dict of metrics for multi-step forecasting
        - plots: Dict of matplotlib figures for visualization
    """
    plots = {}

    # --- single-step forecast ---
    y_pred = single_step_forecasting(
        model=model,
        X_test=X_test,
        sequence_length=sequence_length
    )

    # Handle sequence offset for LSTM models
    if isinstance(model, (KerasModel, LSTMForecaster)):
        y_true = y_test.iloc[sequence_length:]
    else:
        y_true = y_test

    single_step_metrics = calculate_metrics(y_true, y_pred)

    # --- multi-step forecast ---
    y_pred_multi = multi_step_forecasting(
        model=model,
        X_test=X_test,
        forecast_horizon=forecast_horizon,
        sequence_length=sequence_length
    )
    y_true_multi = y_test.iloc[:forecast_horizon]
    multi_step_metrics = calculate_metrics(y_true_multi, y_pred_multi)

    # --- plotting ---
    # 1) single-step
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(y_true.index, y_true, label="Actual")
    ax1.plot(y_pred.index, y_pred, "--", label="Predicted")
    ax1.set_title("Single-Step Forecast vs Actual")
    ax1.set_xlabel("Time"); ax1.set_ylabel(y_test.name)
    ax1.legend(); ax1.grid()
    plots["single_step"] = fig1

    # 2) multi-step
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(y_true_multi.index, y_true_multi, label="Actual")
    ax2.plot(y_pred_multi.index, y_pred_multi, "--", label=f"{forecast_horizon}-Step Forecast")
    ax2.set_title(f"{forecast_horizon}-Step Ahead Forecast vs Actual")
    ax2.set_xlabel("Time"); ax2.set_ylabel(y_test.name)
    ax2.legend(); ax2.grid()
    plots["multi_step"] = fig2

    # 3) error distribution (single-step)
    errors = y_true.values - y_pred.values
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.hist(errors, bins=30, alpha=0.7)
    ax3.axvline(0, color="red", linestyle="--")
    ax3.set_title("Prediction Error Distribution")
    ax3.set_xlabel("Error"); ax3.set_ylabel("Count")
    plots["error_dist"] = fig3

    return single_step_metrics, multi_step_metrics, plots
