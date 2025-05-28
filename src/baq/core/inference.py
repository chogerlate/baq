import numpy as np
import pandas as pd
from keras.models import Model as KerasModel
from baq.data.utils import create_sequences
from baq.models.lstm import LSTMForecaster

def single_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
    sequence_length: int = 1
) -> pd.Series:
    """
    Generate single-step forecasts.
    - For LSTM: Use (samples, seq_len, features)
    - For sklearn/XGB: Use DataFrame 2D

    Args:
      model: LSTMForecaster, sklearn, or XGBoost model
      X_test: DataFrame shape=(T, F)
      sequence_length: length of sliding window (LSTM)
    Returns:
      Series of predictions, index offset to sequence_length  
    """
    if isinstance(model, (KerasModel, LSTMForecaster)):
        # create sequence windows
        X_seq, _ = create_sequences(X_test, pd.Series(np.zeros(len(X_test))), sequence_length)
        # X_seq shape = (T-seq_len, seq_len, F)
        preds = model.predict(X_seq)  # reshape handled in LSTMForecaster.predict()
        # index of y is X_test.index[sequence_length:]
        idx = X_test.index[sequence_length:]
    else:
        preds = model.predict(X_test)
        idx = X_test.index

    return pd.Series(preds, index=idx)

def multi_step_forecasting(
    model: object,
    X_test: pd.DataFrame,
    forecast_horizon: int,
    sequence_length: int = 1
) -> pd.Series:
    """
    Multi-step iterated forecasting.
    
    Args:
        model: LSTMForecaster, sklearn, or XGBoost model
        X_test: Input features DataFrame
        forecast_horizon: Number of steps to forecast
        sequence_length: Length of sequence for LSTM models
        
    Returns:
        Series of predictions with length forecast_horizon
    """
    # copy data to update lag‐features
    X_fore = X_test.copy()
    preds = []

    # buffer for LSTM: keep last `sequence_length` rows
    if isinstance(model, (KerasModel, LSTMForecaster)):
        seq_buffer = X_fore[:sequence_length].to_numpy(dtype=np.float32)

    for step in range(forecast_horizon):
        if isinstance(model, (KerasModel, LSTMForecaster)):
            # LSTM: reshape (1, seq_len, F)
            inp = seq_buffer.reshape(1, sequence_length, -1)
            if isinstance(model, LSTMForecaster):
                p = model.predict(inp)[0]  # predict returns 1D array
            else:
                p = model.predict(inp)[0,0]
        else:
            # sklearn/XGB: use a single row
            row = X_fore.iloc[[step]]
            p = model.predict(row)[0]

        preds.append(p)

        # Update lag‐features in X_fore
        if step < forecast_horizon - 1:
            for col in X_fore.columns:
                if col.endswith(f"_lag_{step+1}"):
                    base = col.rsplit("_lag_",1)[0]
                    next_col = f"{base}_lag_{step+2}"
                    if next_col in X_fore.columns:
                        X_fore.iat[step+1, X_fore.columns.get_loc(next_col)] = p

        # if LSTM, shift buffer and append new value
        if isinstance(model, (KerasModel, LSTMForecaster)):
            seq_buffer = np.vstack([seq_buffer[1:], X_fore.iloc[[step+1]].to_numpy(dtype=np.float32)])

    # index of multi‐step is X_test.index[:forecast_horizon]
    return pd.Series(preds, index=X_test.index[:forecast_horizon])