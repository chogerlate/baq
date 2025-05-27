
import pandas as pd

def single_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions for a single step forecasting model.
    
    This function makes predictions using a trained model on test data
    for a single time step ahead forecasting task.
    
    Args:
        model: Trained model object with a predict method
        X_test: Test features dataframe
        
    Returns:
        pd.Series: Series containing the predicted values
    """
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

def multi_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
    forecast_horizon: int,
) -> pd.Series:
    """
    Generate predictions for a multi-step forecasting model.
    
    This function iteratively predicts multiple steps ahead by using each prediction
    as an input for the next step prediction. It updates the feature matrix after
    each prediction to simulate real forecasting conditions.
    
    Args:
        model: Trained model object
        X_test: Test features
        forecast_horizon: Number of time steps to forecast ahead
        
    Returns:
        pd.Series: Series containing multi-step predictions
    """
    # Make a copy of the test data to avoid modifying the original
    X_forecast = X_test.copy()
    
    # Initialize predictions array
    predictions = []
    
    # Iteratively predict each step
    for step in range(forecast_horizon):
        # Generate prediction for current step
        step_pred = model.predict(X_forecast.iloc[[step]])
        predictions.append(step_pred[0])
        
        # If not the last step, update features for next prediction
        if step < forecast_horizon - 1:
            # Update lag features if they exist in the dataset
            for lag_col in [col for col in X_forecast.columns if col.endswith(f'_lag_{step+1}')]:
                target_col = lag_col.split('_lag_')[0]
                # Find the next lag column to update
                next_lag_col = f"{target_col}_lag_{step+2}"
                if next_lag_col in X_forecast.columns:
                    X_forecast.loc[X_forecast.index[step+1], next_lag_col] = step_pred[0]
    
    return pd.Series(predictions, index=X_test.index[:forecast_horizon])