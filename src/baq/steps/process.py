import pandas as pd
from typing import Tuple
from baq.data.processing import TimeSeriesDataProcessor

def process_train_data(
    data: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.1,
    test_ratio:  float = 0.2,
) -> Tuple[
    pd.DataFrame, pd.Series,   # X_train, y_train
    pd.DataFrame, pd.Series,   # X_val,   y_val
    pd.DataFrame, pd.Series,   # X_test,  y_test
    TimeSeriesDataProcessor
]:
    """
    1) รับ raw DataFrame
    2) ทำ cleaning, feature engineering, scaling, splitting ในตัวเดียว
    3) คืน X_train,y_train, X_val,y_val, X_test,y_test และ processor (เก็บ scalers & encoder)
    """
    # instantiate processor
    processor = TimeSeriesDataProcessor(
        target_col   = target_column,
        train_ratio  = train_ratio,
        val_ratio    = val_ratio,
        test_ratio   = test_ratio,
    )

    # fit + transform in one go
    X_train, y_train, X_val, y_val, X_test, y_test = processor.fit_transform(data)

    return X_train, y_train, X_val, y_val, X_test, y_test, processor
