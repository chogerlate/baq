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
    >>> processor = TimeSeriesDataProcessor(target_col="pm2_5_(μg/m³)")
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
        target_col: str = "pm2_5_(μg/m³)",
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
            - Handles weather code encoding
            - Normalizes column names
        """
        df = df.copy()
        
        # 1) Normalize column names (from old version)
        df.columns = (
            df.columns
            .str.replace(r'\s*\(\)', '', regex=True)
            .str.replace(' ', '_', regex=False)
        )
        
        # 2) Handle datetime index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.set_index('time').sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 3) Drop unwanted columns (from old version)
        to_drop = [
            'carbon_dioxide_ppm',
            'methane_μg/m³', 
            'snowfall_cm',
            'snow_depth_m'
        ]
        df = df.drop(columns=to_drop, errors='ignore')
        
        # 4) Resample to hourly frequency if needed
        if hasattr(df.index, 'freq') and df.index.freq is None:
            df = df.resample('1h').asfreq()
        
        # Infer object dtypes to avoid FutureWarning
        df = df.infer_objects(copy=False)
        
        # 5) Handle weather code encoding (from old version)
        weather_cols = [c for c in df.columns if c.startswith('weather_code')]
        if weather_cols:
            orig = weather_cols[0]
            df['weather_code'] = self.weather_encoder.fit_transform(
                df[orig].astype(str)
            )
            df = df.drop(columns=[orig])
        
        # 6) Handle missing values - only interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=24)  # Short gaps
        
        # 7) Fill remaining gaps with seasonal median (enhanced from old version)
        if len(numeric_cols) > 0:
            df = self._fill_seasonal_median_enhanced(df, numeric_cols)
        
        return df

    def _fill_seasonal_median_enhanced(self, df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """
        Enhanced seasonal median filling that considers year, month, day, and hour.
        
        Args:
            df: DataFrame to fill
            numeric_cols: List of numeric columns to process
            
        Returns:
            DataFrame with filled values
        """
        df = df.copy()
        idx = df.index
        
        # Create temporary time features for grouping
        temp_hour = idx.hour
        temp_day = idx.day  
        temp_month = idx.month
        temp_year = idx.year
        
        for col in numeric_cols:
            missing_mask = df[col].isna()
            if missing_mask.any():
                # Try to fill using same month, day, hour from different years
                for ts in df[missing_mask].index:
                    m, d, h, yr = ts.month, ts.day, ts.hour, ts.year
                    
                    # Look for same month, day, hour in different years
                    same_time_mask = (
                        (temp_month == m) & 
                        (temp_day == d) & 
                        (temp_hour == h) & 
                        (temp_year != yr) & 
                        df[col].notna()
                    )
                    
                    if same_time_mask.any():
                        median_val = df.loc[same_time_mask, col].median()
                        if pd.notna(median_val):
                            df.at[ts, col] = median_val
                            continue
                    
                    # Fallback: use month and hour only (original approach)
                    month_hour_mask = (
                        (temp_month == m) & 
                        (temp_hour == h) & 
                        df[col].notna()
                    )
                    
                    if month_hour_mask.any():
                        median_val = df.loc[month_hour_mask, col].median()
                        if pd.notna(median_val):
                            df.at[ts, col] = median_val
        
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
        - AQI tier classification
        - Weekend/night indicators
        - Cyclical time encoding
        """
        df = df.copy()
        
        # Time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Additional time features from old version
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 20)).astype(int)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Lag features
        for lag in [1, 3, 6, 12, 24]:
            df[f'lag_{lag}h'] = df[self.target_col].shift(lag)
            
        # Rolling statistics - include both old and new windows
        for window in [3, 6, 12, 24]:
            df[f'rolling_mean_{window}h'] = df[self.target_col].rolling(window).mean()
            if window >= 6:  # Only add std for larger windows to avoid noise
                df[f'rolling_std_{window}h'] = df[self.target_col].rolling(window).std()
        
        # AQI tier based on PM2.5 (important feature from old version)
        df['pm2_5_tier'] = df[self.target_col].apply(self._pm25_to_aqi)
            
        # Drop rows with NaN from feature engineering
        df = df.dropna()
        
        return df

    @staticmethod
    def _pm25_to_aqi(x: float) -> int:
        """
        Convert PM2.5 concentration to AQI tier.
        
        Args:
            x: PM2.5 concentration in μg/m³
            
        Returns:
            AQI tier (0-5)
        """
        if x <= 12.0: 
            return 0  # Good
        elif x <= 35.4: 
            return 1  # Moderate
        elif x <= 55.4: 
            return 2  # Unhealthy for Sensitive Groups
        elif x <= 150.4: 
            return 3  # Unhealthy
        elif x <= 250.4: 
            return 4  # Very Unhealthy
        else: 
            return 5  # Hazardous

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

        # Define features and fit scalers - be more inclusive like old version
        all_cols = df.columns.tolist()
        
        # Handle potential target column name variations
        target_variations = [
            self.target_col,
            self.target_col.replace(' ', '_'),
            self.target_col.replace('_', ' '),
            'pm2_5_(μg/m³)',  # Old version format
            'pm2_5 (μg/m³)',  # Current version format
        ]
        
        # Find the actual target column
        actual_target_col = None
        for variation in target_variations:
            if variation in all_cols:
                actual_target_col = variation
                break
        
        if actual_target_col is None:
            raise ValueError(f"Target column not found. Tried: {target_variations}. Available columns: {all_cols}")
        
        # Update target column to actual found column
        self.target_col = actual_target_col
        all_cols.remove(self.target_col)  # Remove target column
        
        # Be more inclusive with feature selection (like old version)
        # Include both numeric and properly encoded categorical features
        feature_cols = []
        for col in all_cols:
            # Include numeric columns
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'int16']:
                feature_cols.append(col)
            # Include binary/categorical columns that are properly encoded
            elif df[col].dtype == 'bool' or (df[col].dtype == 'object' and df[col].nunique() <= 10):
                # Convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        feature_cols.append(col)
                except:
                    pass
        
        self.feature_cols = feature_cols
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
