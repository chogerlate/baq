import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def calculate_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """
    Calculate evaluation metrics for the model.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = 1 - mape

    # Return metrics
    return {
        'mae': round(mae, 6),
        'mape': round(mape, 6),
        'mse': round(mse, 6),
        'rmse': round(rmse, 6),
        'r2': round(r2, 6),
        'accuracy': round(accuracy, 6)
    }