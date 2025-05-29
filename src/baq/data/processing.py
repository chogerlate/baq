"""
Data Processing Module for Bangkok Air Quality Forecasting.

This module provides comprehensive data processing capabilities for PM2.5 forecasting:
- Data cleaning and resampling
- Feature engineering and encoding
- Data splitting and scaling
- Quality validation and checks

The main class TimeSeriesDataProcessor handles the entire preprocessing pipeline
in a scikit-learn compatible way with fit_transform and transform methods.

Example:
    >>> processor = TimeSeriesDataProcessor(target_col="pm2_5 (μg/m³)")
    >>> X_train, y_train, X_val, y_val, X_test, y_test = processor.fit_transform(raw_data)
    >>> X_train.shape
    (1000, 50)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TimeSeriesDataProcessor:
    """
    All-in-one processor for PM2.5 forecasting data preparation.

    This class handles the complete data processing pipeline including:
    1. Data cleaning (resampling, interpolation, seasonal median filling)
    2. Feature encoding (weather codes to integers)
    3. Feature engineering (time features, lags, rolling stats)
    4. Data splitting (train/validation/test)
    5. Feature scaling (MinMax scaling)

    Attributes:
        target_col (str): Name of target column (PM2.5 measurements)
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        feature_cols (List[str]): Names of feature columns after processing
        weather_encoder (LabelEncoder): Encoder for weather codes
        feature_scaler (MinMaxScaler): Scaler for feature columns
        target_scaler (MinMaxScaler): Scaler for target column
    """

    def __init__(
        self,
        target_col: str = "pm2_5 (μg/m³)",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the TimeSeriesDataProcessor.

        Args:
            target_col: Column name for PM2.5 measurements
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.1)
            test_ratio: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        self.target_col = target_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Initialize encoders and scalers
        self.weather_encoder = LabelEncoder()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # Will be set during fit
        self.feature_cols: List[str] = []

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and outliers.

        Args:
            df: Raw input DataFrame

        Returns:
            Cleaned DataFrame

        Note:
            - Resamples to hourly frequency if needed
            - Uses linear interpolation for short gaps
            - Fills longer gaps with seasonal median
        """
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Infer object dtypes to avoid FutureWarning
        df = df.infer_objects(copy=False)
        
        # Handle missing values - only interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=24)  # Short gaps
        
        # Fill remaining gaps with seasonal median (only for numeric columns)
        if len(numeric_cols) > 0:
            seasonal_medians = df[numeric_cols].groupby([df.index.month, df.index.hour]).median()
            
            # Fill NaN values only in numeric columns
            for col in numeric_cols:
                mask = df[col].isna()
                if mask.any():
                    # Create a mapping from (month, hour) to median value
                    for (month, hour), median_val in seasonal_medians[col].items():
                        month_hour_mask = (df.index.month == month) & (df.index.hour == hour) & mask
                        df.loc[month_hour_mask, col] = median_val
        
        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for PM2.5 forecasting.

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with engineered features

        Features created:
        - Time features (hour, day, month, etc.)
        - Lag features (previous values)
        - Rolling statistics (mean, std, etc.)
        - Weather encodings
        """
        df = df.copy()
        
        # Time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Lag features
        for lag in [1, 3, 6, 12, 24]:
            df[f'lag_{lag}h'] = df[self.target_col].shift(lag)
            
        # Rolling statistics
        for window in [6, 12, 24]:
            df[f'rolling_mean_{window}h'] = df[self.target_col].rolling(window).mean()
            df[f'rolling_std_{window}h'] = df[self.target_col].rolling(window).std()
            
        # Drop rows with NaN from feature engineering
        df = df.dropna()
        
        return df

    def _split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Processed DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)

        Note:
            Uses temporal splitting to maintain time series nature
        """
        n = len(df)
        train_idx = int(n * self.train_ratio)
        val_idx = train_idx + int(n * self.val_ratio)
        
        train_df = df[:train_idx]
        val_df = df[train_idx:val_idx]
        test_df = df[val_idx:]
        
        return train_df, val_df, test_df

    def fit_transform(
        self,
        raw: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series
    ]:
        """
        Clean, engineer, split and scale the input DataFrame.

        Args:
            raw: Raw input DataFrame

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)

        Note:
            This method should be used only once for initial training data
        """
        logger.info("Starting data processing pipeline...")
        
        # Clean and engineer features
        df = self._clean(raw)
        logger.info("Data cleaning completed")
        
        df = self._feature_engineer(df)
        logger.info("Feature engineering completed")

        # Split data
        train_df, val_df, test_df = self._split(df)
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Define features and fit scalers - only use numeric columns
        all_cols = df.columns.tolist()
        all_cols.remove(self.target_col)  # Remove target column
        
        # Only keep numeric columns for features
        numeric_feature_cols = []
        for col in all_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_feature_cols.append(col)
        
        self.feature_cols = numeric_feature_cols
        self.feature_scaler.fit(train_df[self.feature_cols])
        self.target_scaler.fit(train_df[[self.target_col]])
        logger.info(f"Scalers fitted on training data with {len(self.feature_cols)} numeric features")
        logger.info(f"Feature columns: {self.feature_cols}")

        # Scale each split
        def _scale(df_: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            """Scale features and target for a given DataFrame."""
            X = pd.DataFrame(
                self.feature_scaler.transform(df_[self.feature_cols]),
                index=df_.index, columns=self.feature_cols
            )
            y = pd.Series(
                self.target_scaler.transform(df_[[self.target_col]]).ravel(),
                index=df_.index, name=self.target_col
            )
            return X, y

        X_train, y_train = _scale(train_df)
        X_val, y_val = _scale(val_df)
        X_test, y_test = _scale(test_df)
        
        logger.info("Data processing pipeline completed successfully")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Process new data using fitted parameters.

        Args:
            raw: Raw input DataFrame

        Returns:
            Processed and scaled features DataFrame

        Note:
            Use this method for new data after fitting on training data
        """
        logger.info("Transforming new data...")
        df = self._clean(raw)
        df = self._feature_engineer(df)
        
        X = pd.DataFrame(
            self.feature_scaler.transform(df[self.feature_cols]),
            index=df.index, columns=self.feature_cols
        )
        logger.info("Data transformation completed")
        return X

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale.

        Args:
            y: Scaled target values

        Returns:
            Target values in original scale
        """
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
