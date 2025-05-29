import pytest
import pandas as pd
import numpy as np
from unittest import mock

# Functions and classes to be tested or mocked
from baq.steps.train import train_model
# from baq.models.lstm import LSTMForecaster # Mocked, so direct import not strictly needed for test execution
# from sklearn.ensemble import RandomForestRegressor # Mocked
# from xgboost import XGBRegressor # Mocked

# Mocked utility
MOCK_METRICS = {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1, 'r2': 0.9, 'mape': 1.0}

@pytest.fixture
def sample_data():
    """Provides sample DataFrames and Series for training and testing."""
    n_rows = 50 # Enough for sequence creation
    n_features = 3
    X_train = pd.DataFrame(np.random.rand(n_rows, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    y_train = pd.Series(np.random.rand(n_rows))
    X_val = pd.DataFrame(np.random.rand(n_rows // 2, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    y_val = pd.Series(np.random.rand(n_rows // 2))
    X_test = pd.DataFrame(np.random.rand(n_rows // 2, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    y_test = pd.Series(np.random.rand(n_rows // 2))
    return X_train, y_train, X_val, y_val, X_test, y_test

@pytest.fixture
def base_config():
    """Provides a base configuration dictionary."""
    return {
        'model_params': {},
        'model_training_params': {},
        'training_config': {'sequence_length': 5} # sequence_length for LSTM
    }

# --- Mocks for external dependencies ---
@mock.patch('baq.steps.train.calculate_metrics', return_value=MOCK_METRICS)
@mock.patch('xgboost.XGBRegressor')
def test_train_model_xgboost(mock_xgb_regressor, mock_calc_metrics, sample_data, base_config):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data
    
    # Configure mock behavior for XGBRegressor
    mock_model_instance = mock.MagicMock()
    mock_xgb_regressor.return_value = mock_model_instance
    mock_model_instance.predict.return_value = np.random.rand(len(y_test))

    model, metrics = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name='xgboost',
        model_params={'n_estimators': 10},
        model_training_params=base_config['model_training_params'],
        training_config=base_config['training_config']
    )

    mock_xgb_regressor.assert_called_once_with(n_estimators=10)
    mock_model_instance.fit.assert_called_once_with(X_train, y_train)
    mock_model_instance.predict.assert_called_once_with(X_test)
    mock_calc_metrics.assert_called_once() # Check if called, args checked by its own tests implicitly
    
    assert model == mock_model_instance
    assert metrics == MOCK_METRICS

@mock.patch('baq.steps.train.calculate_metrics', return_value=MOCK_METRICS)
@mock.patch('sklearn.ensemble.RandomForestRegressor')
def test_train_model_random_forest(mock_rf_regressor, mock_calc_metrics, sample_data, base_config):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    mock_model_instance = mock.MagicMock()
    mock_rf_regressor.return_value = mock_model_instance
    mock_model_instance.predict.return_value = np.random.rand(len(y_test))

    model, metrics = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name='random_forest',
        model_params={'n_estimators': 5},
        model_training_params=base_config['model_training_params'],
        training_config=base_config['training_config']
    )

    mock_rf_regressor.assert_called_once_with(n_estimators=5)
    mock_model_instance.fit.assert_called_once_with(X_train, y_train)
    mock_model_instance.predict.assert_called_once_with(X_test)
    mock_calc_metrics.assert_called_once()
    
    assert model == mock_model_instance
    assert metrics == MOCK_METRICS


@mock.patch('baq.steps.train.calculate_metrics', return_value=MOCK_METRICS)
@mock.patch('baq.steps.train.LSTMForecaster') # Mock the LSTMForecaster class
@mock.patch('baq.steps.train.create_sequences') # Mock create_sequences utility
def test_train_model_lstm(mock_create_sequences, mock_lstm_forecaster, mock_calc_metrics, sample_data, base_config):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data
    seq_len = base_config['training_config']['sequence_length']

    # Configure mock for create_sequences
    # It should return sequences of appropriate shape
    # (num_samples, sequence_length, num_features) for X, (num_samples,) for y
    mock_X_tr_seq = np.random.rand(len(X_train) - seq_len +1, seq_len, X_train.shape[1]) # Corrected length calculation
    mock_y_tr_seq = np.random.rand(len(X_train) - seq_len +1) # Corrected length calculation
    mock_X_val_seq = np.random.rand(len(X_val) - seq_len +1, seq_len, X_val.shape[1]) # Corrected length calculation
    mock_y_val_seq = np.random.rand(len(X_val) - seq_len +1) # Corrected length calculation
    mock_X_te_seq = np.random.rand(len(X_test) - seq_len +1, seq_len, X_test.shape[1]) # Corrected length calculation
    mock_y_te_seq = np.random.rand(len(X_test) - seq_len +1) # Corrected length calculation (this is what calculate_metrics will use for y_true)


    mock_create_sequences.side_effect = [
        (mock_X_tr_seq, mock_y_tr_seq),
        (mock_X_val_seq, mock_y_val_seq),
        (mock_X_te_seq, mock_y_te_seq)
    ]

    # Configure mock for LSTMForecaster
    mock_model_instance = mock.MagicMock()
    mock_lstm_forecaster.return_value = mock_model_instance
    # The predict method of LSTMForecaster should return predictions matching y_te_seq length
    mock_model_instance.predict.return_value = np.random.rand(len(mock_y_te_seq))


    lstm_model_params = {'lstm_units': (64,), 'dropout_rate': 0.1, 'learning_rate': 0.01}
    lstm_training_params = {'epochs': 1, 'batch_size': 16, 'early_stopping_patience': 3, 'reduce_lr_patience': 2}
    
    model, metrics = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name='lstm',
        model_params=lstm_model_params,
        model_training_params=lstm_training_params,
        training_config=base_config['training_config']
    )

    # Check create_sequences calls
    assert mock_create_sequences.call_count == 3
    mock_create_sequences.assert_any_call(X_train, y_train, seq_len)
    mock_create_sequences.assert_any_call(X_val, y_val, seq_len)
    mock_create_sequences.assert_any_call(X_test, y_test, seq_len)

    # Check LSTMForecaster instantiation
    expected_lstm_constructor_params = {
        "input_shape": (seq_len, X_train.shape[1]),
        "lstm_units": lstm_model_params['lstm_units'],
        "dropout_rate": lstm_model_params['dropout_rate'],
        "learning_rate": lstm_model_params['learning_rate'],
        "checkpoint_path": "best_lstm.h5", # Default from train_model
        "early_stopping_patience": lstm_training_params['early_stopping_patience'],
        "reduce_lr_patience": lstm_training_params['reduce_lr_patience']
    }
    mock_lstm_forecaster.assert_called_once_with(**expected_lstm_constructor_params)

    # Check fit call on LSTM model instance
    mock_model_instance.fit.assert_called_once_with(
        mock_X_tr_seq, mock_y_tr_seq,
        validation_data=(mock_X_val_seq, mock_y_val_seq),
        epochs=lstm_training_params['epochs'],
        batch_size=lstm_training_params['batch_size'],
        shuffle=False, # Default from train_model
        verbose=1      # Default from train_model
    )

    # Check predict call
    mock_model_instance.predict.assert_called_once_with(mock_X_te_seq)
    
    # Check metrics calculation (y_true should be y_te_seq)
    # Retrieve the actual y_true passed to calculate_metrics
    actual_calc_metrics_call = mock_calc_metrics.call_args
    assert actual_calc_metrics_call is not None
    passed_y_true, _ = actual_calc_metrics_call[0] # Get positional arguments
    np.testing.assert_array_equal(passed_y_true, mock_y_te_seq)

    assert model == mock_model_instance
    assert metrics == MOCK_METRICS

def test_train_model_unsupported(sample_data, base_config):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data
    with pytest.raises(ValueError, match="Unsupported model: unsupported_model_type"):
        train_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_name='unsupported_model_type',
            model_params=base_config['model_params'],
            model_training_params=base_config['model_training_params'],
            training_config=base_config['training_config']
        )

# To run: pytest tests/unit/steps/test_train.py
# (or `uv run pytest tests/unit/steps/test_train.py`)
