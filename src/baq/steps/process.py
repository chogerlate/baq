"""
This module contains the code for processing the data.
- Handles data cleaning, transformation, and feature engineering
- Prepares raw data for model training and inference
- Implements preprocessing pipelines for consistent data handling
- Manages categorical encoding, numerical scaling, and missing value imputation
- Ensures data is properly formatted to be fed into the model
"""
import pandas as pd
from baq.data.processing import TimeSeriesPreprocessor, create_time_series_processor

# 2. Feature Engineering
def create_features(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Create features.
    """
    # Create time-based features (always available)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek >= 5
    
    # Create lag features for PM2.5 (only use lags that would be available at prediction time)
    # For real-time forecasting, we'd have access to historical PM2.5 values
    for lag in [1, 3, 6, 12, 24]:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    # Create rolling statistics for key weather variables that can be obtained from weather forecasts
    forecast_vars = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 
        'wind_speed_10m', 'wind_direction_10m', 'pressure_msl'
    ]
    
    for var in forecast_vars:
        # Use smaller windows that are more realistic for forecasting
        for window in [3, 6, 12]:
            df[f'{var}_rolling_mean_{window}'] = df[var].rolling(window=window).mean()
    
    # Drop rows with NaN values created by lag features
    df.dropna(inplace=True)
    
    return df

# 3. Feature Selection
def select_features(
    df: pd.DataFrame,
    forecast_horizon: int,
    target_column: str,
) -> list:
    """
    Select features for model training and prediction based on forecast horizon.
    
    Args:
        df: DataFrame containing all potential features
        forecast_horizon: Number of time steps to forecast ahead
        target_column: Name of the target variable column
        
    Returns:
        list: Selected feature names that would be available at prediction time
    """
    # Define features to exclude
    exclude_features = [target_column, 'target']  # Target variables
    
    # Weather variables that can be obtained from weather forecasts
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'pressure_msl'
    ]
    
    # Time features (always available)
    time_features = ['hour', 'day', 'month', 'day_of_week', 'is_weekend']
    
    # Lag features (only those available at prediction time)
    lag_features = [f for f in df.columns if f.startswith(f'{target_column}_lag_') and 
                   int(f.split('_')[-1]) >= forecast_horizon]
    
    # Rolling statistics of weather variables (based on weather forecasts)
    rolling_features = [f for f in df.columns if 'rolling_mean' in f and
                       any(var in f for var in weather_features)]
    
    # Weather code dummy variables (can be obtained from weather forecasts)
    weather_code_features = [col for col in df.columns if col.startswith('weather_code_')]
    
    # Combine all selected features
    selected_features = (weather_features + time_features + 
                        lag_features + rolling_features + weather_code_features)
    
    # Ensure all selected features are in the dataframe and not in excluded list
    selected_features = [f for f in selected_features if f in df.columns and f not in exclude_features]
    
    return selected_features

def convert_to_datetime(
    df: pd.DataFrame,
    datetime_column: str,
) -> pd.DataFrame:
    """
    Convert the time column to a datetime and set it as the index.
    """
    # Convert the data to a pandas DataFrame
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df.set_index(datetime_column, inplace=True)
    df.sort_index(inplace=True)
    return df

def process_target_variable(
    df: pd.DataFrame,
    target_column: str,
    forecast_horizon: int,
) -> pd.DataFrame:
    """
    Process the target variable.
    """
    # For 1-hour ahead forecasting, shift the target back by 1
    df['target'] = df[target_column].shift(-forecast_horizon)
    df.dropna(inplace=True)
    return df

def process_train_data(
    data: pd.DataFrame,
    target_column: str,
    forecast_horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, TimeSeriesPreprocessor]:
    """
    Process the data and split into train and test sets.
    
    Args:
        data: The data to process
        target_column: The column to predict
        forecast_horizon: The number of time steps to forecast ahead
        
    Returns:
        tuple: X_train, y_train, X_test, y_test, processor
    """
    # Copy the data
    df = data.copy()

    # Convert the time column to a datetime and set it as the index
    df = convert_to_datetime(
        df=df,
        datetime_column='time'
    )
    
    # Handle missing values using forward fill (more realistic than mean)
    df['carbon_dioxide'] = df['carbon_dioxide'].ffill()

    # Create features
    df = create_features(
        df=df,
        target_column=target_column
    )

    # Process the target variable
    df = process_target_variable(
        df=df,
        target_column=target_column,
        forecast_horizon=forecast_horizon
    )
    
    # Select features
    features = select_features(
        df=df,
        forecast_horizon=forecast_horizon,
        target_column=target_column
    )
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    # Create feature and target variables for train and test sets
    X_train = df_train[features]
    y_train = df_train['target']
    X_test = df_test[features]
    y_test = df_test['target']
    
    # Define feature types for preprocessing
    numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist() 
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Define cyclical features (time-based)
    cyclical_features = ['hour', 'day_of_week', 'month'] 
    cyclical_features = [f for f in cyclical_features if f in features]
    
    # Features that might benefit from log transformation
    log_transform_candidates = ['precipitation', 'temperature_2m', 'relative_humidity_2m']
    log_transform_features = [f for f in log_transform_candidates if f in numerical_features]
    
    print(f"Feature counts - Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}, "
          f"Cyclical: {len(cyclical_features)}")
    
    # Create and fit our processor
    processor = create_time_series_processor(
        df=X_train,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        cyclical_features=cyclical_features,
        log_transform_features=log_transform_features
    )
   

    # Transform the data
    X_train_processed = processor.fit_transform(X_train)
    X_test_processed = processor.transform(X_test)

    return X_train_processed, y_train, X_test_processed, y_test, processor