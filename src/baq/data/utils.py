import numpy as np
import pandas as pd
from typing import Tuple

def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM
    """
    feature_array = X.values
    target_array  = y.values
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(feature_array[i : i + sequence_length])
        ys.append(target_array[i + sequence_length])
    return np.array(Xs), np.array(ys)
