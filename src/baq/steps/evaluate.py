"""
This module contains the code for evaluating the model.
- In this case, we develop forecasting models, so the evaluation is done by comparing the predicted values with the actual values.
- The evaluation metrics are the mean absolute error, the mean squared error, and the R-squared score.
- The evaluation is done on the test set.
"""
import pandas as pd
from baq.core.evaluation import calculate_metrics
from baq.core.inference import single_step_forecasting, multi_step_forecasting

def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    forecast_horizon: int,
) -> tuple:
    """
    Evaluate the model on test data and generate performance visualizations.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        forecast_horizon: Number of steps to forecast ahead
        
    Returns:
        tuple: Tuple containing single-step metrics, multi-step metrics, and visualization plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate predictions
    y_pred = single_step_forecasting(model, X_test)
    
    # Calculate metrics
    single_step_metrics = calculate_metrics(y_test, y_pred)
    
    # Calculate multi-step metrics
    y_pred_multi_step = multi_step_forecasting(model, X_test, forecast_horizon)
    # Use only the first forecast_horizon samples from y_test for multi-step evaluation
    multi_step_metrics = calculate_metrics(y_test[:forecast_horizon], y_pred_multi_step)
    
    # Create visualization plots
    plots = {}
    
    # Single-step forecast visualization
    fig_single, ax_single = plt.subplots(figsize=(12, 6))
    ax_single.plot(y_test.index, y_test, label='Actual', color='blue')
    ax_single.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
    ax_single.set_title('Single-Step Forecast vs Actual Values')
    ax_single.set_xlabel('Time')
    ax_single.set_ylabel('Value')
    ax_single.legend()
    ax_single.grid(True)
    plots['single_step_forecast'] = fig_single
    
    # Multi-step forecast visualization
    fig_multi, ax_multi = plt.subplots(figsize=(12, 6))
    ax_multi.plot(y_test[:forecast_horizon].index, y_test[:forecast_horizon], 
                 label='Actual', color='blue')
    ax_multi.plot(y_test[:forecast_horizon].index, y_pred_multi_step, 
                 label='Multi-step Forecast', color='green', linestyle='--')
    ax_multi.set_title(f'{forecast_horizon}-Step Ahead Forecast vs Actual Values')
    ax_multi.set_xlabel('Time')
    ax_multi.set_ylabel('Value')
    ax_multi.legend()
    ax_multi.grid(True)
    plots['multi_step_forecast'] = fig_multi
    
    # Error distribution plot
    fig_error, ax_error = plt.subplots(figsize=(10, 6))
    errors = y_test - y_pred
    ax_error.hist(errors, bins=30, alpha=0.7)
    ax_error.axvline(x=0, color='red', linestyle='--')
    ax_error.set_title('Prediction Error Distribution')
    ax_error.set_xlabel('Error')
    ax_error.set_ylabel('Frequency')
    plots['error_distribution'] = fig_error
    
    return single_step_metrics, multi_step_metrics, plots
