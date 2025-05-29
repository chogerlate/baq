import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Import the class to be tested
# This will work because of tests/context.py
from baq.data.processing import TimeSeriesDataProcessor

TARGET_COL = 'pm2_5' # Simplified for tests

@pytest.fixture
def sample_raw_data_dict():
    """Provides a dictionary for creating a sample raw DataFrame."""
    return {
        'time': pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
            '2023-01-01 03:00:00', '2023-01-01 04:00:00', '2023-01-01 05:00:00',
            '2023-01-01 06:00:00', '2023-01-01 07:00:00', '2023-01-01 08:00:00',
            '2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00' 
        ]),
        TARGET_COL: [10, 12, np.nan, 13, 16, 18, np.nan, 22, 25, 20, 19, 21],
        'temp': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20],
        'humidity': [50, 52, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62],
        'weather_code_ (text)': ['clear', 'clear', 'cloudy', 'cloudy', 'rain', 'rain', 
                                 'clear', 'cloudy', 'rain', 'clear', 'cloudy', 'rain'],
        'carbon_dioxide_ppm': [400] * 12, # To be dropped
    }

@pytest.fixture
def sample_raw_data(sample_raw_data_dict):
    """Provides a sample raw DataFrame for testing."""
    return pd.DataFrame(sample_raw_data_dict)

@pytest.fixture
def default_processor():
    """Returns a TimeSeriesDataProcessor with default initialization."""
    return TimeSeriesDataProcessor(target_col=TARGET_COL, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

def test_processor_initialization():
    processor = TimeSeriesDataProcessor(
        target_col='pm2.5_level',
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_state=123
    )
    assert processor.target_col == 'pm2.5_level'
    assert processor.train_ratio == 0.5
    assert processor.val_ratio == 0.25
    assert processor.test_ratio == 0.25
    assert processor.random_state == 123
    assert isinstance(processor.weather_encoder, LabelEncoder)
    assert isinstance(processor.feature_scaler, MinMaxScaler)
    assert isinstance(processor.target_scaler, MinMaxScaler)

def test_processor_initialization_invalid_ratios():
    with pytest.raises(ValueError, match="Train, validation, and test ratios must sum to 1.0"):
        TimeSeriesDataProcessor(train_ratio=0.7, val_ratio=0.2, test_ratio=0.2) # Sums to 1.1

def test_pm25_to_aqi_static_method():
    assert TimeSeriesDataProcessor._pm25_to_aqi(10) == 0  # Good
    assert TimeSeriesDataProcessor._pm25_to_aqi(30) == 1  # Moderate
    assert TimeSeriesDataProcessor._pm25_to_aqi(50) == 2  # Unhealthy for Sensitive
    assert TimeSeriesDataProcessor._pm25_to_aqi(100) == 3 # Unhealthy
    assert TimeSeriesDataProcessor._pm25_to_aqi(200) == 4 # Very Unhealthy
    assert TimeSeriesDataProcessor._pm25_to_aqi(300) == 5 # Hazardous

def test_clean_data(default_processor, sample_raw_data):
    cleaned_df = default_processor._clean(sample_raw_data.copy()) # Use copy

    # Check datetime index
    assert isinstance(cleaned_df.index, pd.DatetimeIndex)
    assert cleaned_df.index.name == 'time'
    assert cleaned_df.index.is_monotonic_increasing

    # Check dropped columns
    assert 'carbon_dioxide_ppm' not in cleaned_df.columns
    
    # Check column name normalization
    assert 'weather_code' in cleaned_df.columns # from 'weather_code_ (text)'
    assert 'pm2_5' in cleaned_df.columns # from TARGET_COL

    # Check weather_code encoding
    assert pd.api.types.is_integer_dtype(cleaned_df['weather_code'])
    assert default_processor.weather_encoder.classes_ is not None # Encoder should be fitted

    # Check NaN interpolation for target_col (pm2_5)
    # Original: [10, 12, np.nan, 13, 16, 18, np.nan, 22, 25, 20, 19, 21]
    # After linear interpolation (limit 24):
    # 2nd NaN (index 6) should be filled: (18+22)/2 = 20
    # 1st NaN (index 2) should be filled: (12+13)/2 = 12.5
    assert not cleaned_df[TARGET_COL].isnull().any(), "NaNs should be filled in target column"
    assert cleaned_df[TARGET_COL].iloc[2] == 12.5 
    assert cleaned_df[TARGET_COL].iloc[6] == 20.0
    
    # Check other numeric columns are interpolated
    assert not cleaned_df['temp'].isnull().any()
    
    # Check resampling (if original data wasn't hourly) - this sample is hourly
    # If data was, e.g., daily, this would test resampling logic.
    # For this sample, freq should be 'H' or similar.
    if hasattr(cleaned_df.index, 'freqstr'): # pandas < 2.2
        assert cleaned_df.index.freqstr == 'H'
    elif hasattr(cleaned_df.index, 'inferred_freq'): # pandas >= 2.2
         assert cleaned_df.index.inferred_freq == 'H'


def test_feature_engineer(default_processor, sample_raw_data):
    # Need to clean data first as _feature_engineer expects it
    # And manually set target_col as it's set in fit_transform
    default_processor.target_col = TARGET_COL 
    cleaned_df = default_processor._clean(sample_raw_data.copy())
    
    # Simulate that weather_encoder was fit
    default_processor.weather_encoder.fit(sample_raw_data['weather_code_ (text)'].astype(str))
    cleaned_df['weather_code'] = default_processor.weather_encoder.transform(cleaned_df['weather_code'].astype(str))

    features_df = default_processor._feature_engineer(cleaned_df)

    # Check for time features
    assert 'hour' in features_df.columns
    assert 'day' in features_df.columns
    assert 'sin_hour' in features_df.columns
    assert 'is_weekend' in features_df.columns

    # Check for lag features (e.g., lag_1h for TARGET_COL)
    assert f'lag_1h' in features_df.columns
    assert features_df[f'lag_1h'].isnull().sum() == 0 # NaNs from lags should be dropped

    # Check for rolling features
    assert f'rolling_mean_3h' in features_df.columns
    assert features_df[f'rolling_mean_3h'].isnull().sum() == 0

    # Check AQI tier
    assert 'pm2_5_tier' in features_df.columns
    assert features_df['pm2_5_tier'].apply(lambda x: 0 <= x <= 5).all()

    # Check that rows with NaNs from feature engineering are dropped
    # The number of rows will be less than cleaned_df due to lags/rolling windows
    assert len(features_df) < len(cleaned_df)
    assert not features_df.isnull().any().any()


def test_split_data(default_processor, sample_raw_data):
    # Create a dummy DataFrame that's already processed (cleaned and feature engineered)
    # For simplicity, let's just use a range index and a single column
    n_rows = 100
    processed_dummy_df = pd.DataFrame({'feature': range(n_rows)})
    
    processor_for_split = TimeSeriesDataProcessor(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = processor_for_split._split(processed_dummy_df)
    
    assert len(train_df) == int(n_rows * 0.7)
    assert len(val_df) == int(n_rows * 0.15)
    assert len(test_df) == int(n_rows * 0.15)
    
    # Check for temporal order (indices should be contiguous)
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()


def test_fit_transform(default_processor, sample_raw_data):
    # Use a small subset of data for faster test
    data_subset = sample_raw_data.head(30).copy() # Ensure enough data for lags/splits
    
    # Make data longer to avoid issues with splitting after dropna in feature engineering
    # The sample_raw_data only has 12 rows. After lags (max 24h) it becomes empty.
    # We need at least 24 + (train+val+test splits) rows.
    # Let's create more data for fit_transform
    
    extended_data_list = []
    base_time = pd.to_datetime('2023-01-01 00:00:00')
    for i in range(100): # Create 100 hours of data
        extended_data_list.append({
            'time': base_time + pd.Timedelta(hours=i),
            TARGET_COL: 10 + i*0.1 + np.random.randn()*0.5, # some trend and noise
            'temp': 15 + i*0.05,
            'humidity': 50 + i*0.1,
            'weather_code_ (text)': np.random.choice(['clear', 'cloudy', 'rain']),
            'carbon_dioxide_ppm': 400
        })
    extended_raw_data = pd.DataFrame(extended_data_list)


    X_train, y_train, X_val, y_val, X_test, y_test = default_processor.fit_transform(extended_raw_data)

    # Check shapes (number of rows will vary due to dropna in _feature_engineer)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    # Check that total rows roughly match split ratios after feature engineering drops
    # This is an indirect check as exact numbers are hard due to dropna
    total_processed_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    # Max lag is 24, so approx 24 rows are dropped by feature engineering.
    # Initial 100 rows - 24 = 76 rows approx.
    assert total_processed_rows > (len(extended_raw_data) - 24 - 5) # Allow some slack
    assert total_processed_rows < len(extended_raw_data)

    # Check types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    # ... and for val/test

    # Check scaling (MinMaxScaler means values are between 0 and 1, or close due to float precision)
    # Need to handle cases where a feature might be constant after split.
    for col in X_train.columns:
        if not X_train[col].nunique() <= 1: # Only check if not constant
             assert X_train[col].min() >= -1e-6 # Allow for small float inaccuracies
             assert X_train[col].max() <= 1 + 1e-6
    if not y_train.nunique() <= 1:
        assert y_train.min() >= -1e-6
        assert y_train.max() <= 1 + 1e-6
    
    # Check that scalers and encoder are fitted
    assert default_processor.feature_scaler.n_features_in_ > 0
    assert default_processor.target_scaler.n_features_in_ > 0 # Should be 1
    assert hasattr(default_processor.weather_encoder, 'classes_')
    assert len(default_processor.feature_cols) > 0
    assert default_processor.target_col == TARGET_COL # Should be updated if original was different

    # Check that feature_cols are subset of X_train.columns
    assert all(fc in X_train.columns for fc in default_processor.feature_cols)


def test_transform(default_processor, sample_raw_data):
    # Fit the processor first
    extended_data_list = []
    base_time = pd.to_datetime('2023-01-01 00:00:00')
    for i in range(100): # Create 100 hours of data
        extended_data_list.append({
            'time': base_time + pd.Timedelta(hours=i),
            TARGET_COL: 10 + i*0.1,
            'temp': 15 + i*0.05,
            'humidity': 50 + i*0.1,
            'weather_code_ (text)': np.random.choice(['clear', 'cloudy', 'rain']),
        })
    train_data = pd.DataFrame(extended_data_list)
    default_processor.fit_transform(train_data.copy()) # Use copy

    # Now use transform on new data (e.g., a subset of original sample_raw_data)
    # Ensure it has enough rows for feature engineering to not make it empty
    new_data_list = []
    new_base_time = pd.to_datetime('2023-01-05 00:00:00') # Later data
    for i in range(30): # Create 30 hours for test transform
         new_data_list.append({
            'time': new_base_time + pd.Timedelta(hours=i),
            TARGET_COL: 20 + i*0.1, # Different values
            'temp': 20,
            'humidity': 60,
            'weather_code_ (text)': np.random.choice(['clear', 'cloudy', 'rain']),
        })
    new_raw_data = pd.DataFrame(new_data_list)
    
    X_transformed = default_processor.transform(new_raw_data.copy())

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] <= len(new_raw_data) # Rows dropped by feature eng
    assert X_transformed.shape[1] == len(default_processor.feature_cols)
    
    # Check scaling (values should be roughly within [0,1] if they fall within fitted range)
    # Some might be outside if new data is outside original training range.
    # This is expected for transform.
    assert not X_transformed.isnull().any().any()


def test_inverse_transform_target(default_processor, sample_raw_data):
    extended_data_list = []
    base_time = pd.to_datetime('2023-01-01 00:00:00')
    for i in range(100):
        extended_data_list.append({
            'time': base_time + pd.Timedelta(hours=i),
            TARGET_COL: 10 + i*0.1 + np.random.randn()*0.5,
            'temp': 15, 'humidity': 50, 'weather_code_ (text)': 'clear'
        })
    train_data = pd.DataFrame(extended_data_list)
    _, y_train, _, _, _, _ = default_processor.fit_transform(train_data.copy())

    # Inverse transform y_train
    y_train_original_scale = default_processor.inverse_transform_target(y_train.values)
    
    # Compare with the original target values (after cleaning and feature engineering)
    # This is a bit tricky as y_train corresponds to target after _clean and _feature_engineer,
    # and then _split. We need the target values from the *training split* of the *processed dataframe*
    # *before* it was scaled.
    
    # Let's re-engineer to get the unscaled training target
    df_cleaned = default_processor._clean(train_data.copy())
    df_featured = default_processor._feature_engineer(df_cleaned) # This drops NaNs
    
    # Manually split to get the unscaled training target that corresponds to y_train
    n = len(df_featured)
    train_idx = int(n * default_processor.train_ratio)
    # val_idx = train_idx + int(n * default_processor.val_ratio) # Not needed for y_train part
    
    unscaled_y_train_expected = df_featured[TARGET_COL][:train_idx]

    assert isinstance(y_train_original_scale, np.ndarray)
    assert y_train_original_scale.shape[0] == y_train.shape[0]
    
    # Check that the inverse transformed values are close to the original (unscaled) y_train values
    # The lengths might differ slightly if feature engineering caused different numbers of NaNs
    # at the beginning of the *original* train_data versus what became y_train.
    # The y_train is already aligned with X_train after all processing.
    # So, unscaled_y_train_expected should also be based on the same index as y_train.
    
    # Re-get y_train from the unscaled data based on X_train's index
    # This is the y that was scaled to become y_train
    y_train_unscaled_source = df_featured.loc[y_train.index, TARGET_COL]

    np.testing.assert_allclose(y_train_original_scale, y_train_unscaled_source.values, rtol=1e-5)
