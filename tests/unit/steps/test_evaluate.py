import pytest
import pandas as pd
import numpy as np
from unittest import mock
import matplotlib.pyplot as plt # Used for type hinting and mocking Figure

# Function to be tested
from baq.steps.evaluate import evaluate_model

# Objects that will be mocked
# from baq.core.inference import single_step_forecasting, multi_step_forecasting # Mocked
# from baq.core.evaluation import calculate_metrics # Mocked
# from baq.models.lstm import LSTMForecaster # For isinstance checks
# from keras.models import Model as KerasModel # For isinstance checks


# --- Mock Data and Configurations ---
MOCK_SINGLE_METRICS = {'mae': 0.1, 'mse': 0.01}
MOCK_MULTI_METRICS = {'mae': 0.2, 'mse': 0.04}
FORECAST_HORIZON = 5
SEQUENCE_LENGTH = 3

@pytest.fixture
def sample_test_data():
    """Provides sample X_test and y_test DataFrames/Series."""
    n_rows = 20
    n_features = 2
    X_test = pd.DataFrame(
        np.random.rand(n_rows, n_features), 
        columns=[f'feat_{i}' for i in range(n_features)],
        index=pd.date_range(start='2023-01-01', periods=n_rows, freq='H')
    )
    y_test = pd.Series(
        np.random.rand(n_rows), 
        index=X_test.index, 
        name='target'
    )
    return X_test, y_test

@pytest.fixture
def mock_dependencies():
    """Central fixture to mock all external dependencies for evaluate_model."""
    with mock.patch('baq.steps.evaluate.single_step_forecasting') as mock_single_step, \
         mock.patch('baq.steps.evaluate.multi_step_forecasting') as mock_multi_step, \
         mock.patch('baq.steps.evaluate.calculate_metrics') as mock_calc_metrics, \
         mock.patch('matplotlib.pyplot.subplots') as mock_subplots:

        # Configure single_step_forecasting mock
        # Needs to return a Series with an index that matches y_true after slicing
        # y_true for single step is y_test.iloc[SEQUENCE_LENGTH:] if LSTM, else y_test
        # So, its index should match that.
        # Let's make its length consistent with an example y_test of length 20 and SEQUENCE_LENGTH 3
        # y_true_single_len = 20 - SEQUENCE_LENGTH = 17
        mock_y_pred_single = pd.Series(np.random.rand(20 - SEQUENCE_LENGTH)) # Placeholder length
        mock_single_step.return_value = mock_y_pred_single
        
        # Configure multi_step_forecasting mock
        # Returns predictions for forecast_horizon
        mock_y_pred_multi = pd.Series(np.random.rand(FORECAST_HORIZON))
        mock_multi_step.return_value = mock_y_pred_multi

        # Configure calculate_metrics mock
        # It will be called twice, return different values for single and multi
        mock_calc_metrics.side_effect = [MOCK_SINGLE_METRICS, MOCK_MULTI_METRICS]

        # Configure subplots mock
        # It should return a figure and an axes object
        mock_fig = mock.MagicMock(spec=plt.Figure)
        mock_ax = mock.MagicMock(spec=plt.Axes)
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        yield {
            'single_step': mock_single_step,
            'multi_step': mock_multi_step,
            'calc_metrics': mock_calc_metrics,
            'subplots': mock_subplots,
            'mock_fig': mock_fig,
            'mock_ax': mock_ax,
            'mock_y_pred_single': mock_y_pred_single, # So we can set its index in tests
            'mock_y_pred_multi': mock_y_pred_multi,   # So we can set its index in tests
        }


# --- Test Cases ---

def run_evaluate_model_and_asserts(mock_model_type, sample_test_data, mock_dependencies):
    X_test, y_test = sample_test_data
    
    # Create a mock model instance that matches the type for isinstance checks
    if mock_model_type == 'lstm':
        # For isinstance check, creating a mock that "is" an LSTMForecaster
        # This requires LSTMForecaster to be available or mock-patched at baq.steps.evaluate
        with mock.patch('baq.steps.evaluate.LSTMForecaster', spec=True) as MockLSTMClass:
            MockLSTMClass.__name__ = 'LSTMForecaster' # For mock identification
            mock_model = MockLSTMClass()
    elif mock_model_type == 'keras':
        # Similarly for KerasModel
        with mock.patch('baq.steps.evaluate.KerasModel', spec=True) as MockKerasClass:
            MockKerasClass.__name__ = 'KerasModel'
            mock_model = MockKerasClass()
    else: # Sklearn-like
        mock_model = mock.MagicMock() # Generic mock for sklearn models
        # To ensure it doesn't pass isinstance checks for LSTM/Keras if those are broadly mocked
        mock_model.__class__.__name__ = "MockSklearnModel"


    # Adjust mock_y_pred_single index based on model type and y_test
    if mock_model_type in ['lstm', 'keras']:
        mock_dependencies['mock_y_pred_single'].index = y_test.index[SEQUENCE_LENGTH:]
    else:
        mock_dependencies['mock_y_pred_single'].index = y_test.index
    
    # Adjust mock_y_pred_multi index
    mock_dependencies['mock_y_pred_multi'].index = y_test.index[:FORECAST_HORIZON]


    single_metrics, multi_metrics, plots = evaluate_model(
        model=mock_model,
        X_test=X_test,
        y_test=y_test,
        forecast_horizon=FORECAST_HORIZON,
        sequence_length=SEQUENCE_LENGTH
    )

    # 1. Assert forecasting functions were called correctly
    mock_dependencies['single_step'].assert_called_once_with(
        model=mock_model, X_test=X_test, sequence_length=SEQUENCE_LENGTH
    )
    mock_dependencies['multi_step'].assert_called_once_with(
        model=mock_model, X_test=X_test, forecast_horizon=FORECAST_HORIZON, sequence_length=SEQUENCE_LENGTH
    )

    # 2. Assert calculate_metrics was called correctly (twice)
    assert mock_dependencies['calc_metrics'].call_count == 2
    
    # Call 1 (single-step)
    call_args_single = mock_dependencies['calc_metrics'].call_args_list[0][0]
    y_true_single_expected = y_test.iloc[SEQUENCE_LENGTH:] if mock_model_type in ['lstm', 'keras'] else y_test
    pd.testing.assert_series_equal(call_args_single[0], y_true_single_expected, check_dtype=False)
    pd.testing.assert_series_equal(call_args_single[1], mock_dependencies['mock_y_pred_single'], check_dtype=False)
    
    # Call 2 (multi-step)
    call_args_multi = mock_dependencies['calc_metrics'].call_args_list[1][0]
    y_true_multi_expected = y_test.iloc[:FORECAST_HORIZON]
    pd.testing.assert_series_equal(call_args_multi[0], y_true_multi_expected, check_dtype=False)
    pd.testing.assert_series_equal(call_args_multi[1], mock_dependencies['mock_y_pred_multi'], check_dtype=False)

    # 3. Assert returned metrics are correct
    assert single_metrics == MOCK_SINGLE_METRICS
    assert multi_metrics == MOCK_MULTI_METRICS

    # 4. Assert plotting occurred
    assert mock_dependencies['subplots'].call_count == 3 # For single-step, multi-step, error_dist
    
    # Check plot dictionary
    assert "single_step" in plots
    assert "multi_step" in plots
    assert "error_dist" in plots
    assert isinstance(plots["single_step"], mock.MagicMock) # Check it's the mocked Figure
    assert plots["single_step"] == mock_dependencies['mock_fig']
    

def test_evaluate_model_lstm_type(sample_test_data, mock_dependencies):
    """Test evaluate_model with a model type that triggers sequence handling (e.g., LSTM)."""
    # The run_evaluate_model_and_asserts function handles patching for isinstance checks locally
    run_evaluate_model_and_asserts('lstm', sample_test_data, mock_dependencies)


def test_evaluate_model_sklearn_type(sample_test_data, mock_dependencies):
    """Test evaluate_model with a model type that does not trigger sequence handling (e.g., sklearn)."""
    run_evaluate_model_and_asserts('sklearn', sample_test_data, mock_dependencies)

# To run: pytest tests/unit/steps/test_evaluate.py
# (or `uv run pytest tests/unit/steps/test_evaluate.py`)
