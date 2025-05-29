import pytest
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the function to be tested
# This will work because of tests/context.py
from baq.core.evaluation import calculate_metrics

def test_calculate_metrics_basic():
    """Test calculate_metrics with simple, known values."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5]) # Perfect prediction
    
    expected_metrics = {
        'mae': 0.0,
        'mse': 0.0,
        'rmse': 0.0,
        'r2': 1.0,
        'mape': 0.0
    }
    
    actual_metrics = calculate_metrics(y_true, y_pred)
    
    for key in expected_metrics:
        assert np.isclose(actual_metrics[key], expected_metrics[key]), f"Metric {key} failed for perfect prediction"

def test_calculate_metrics_imperfect():
    """Test calculate_metrics with imperfect predictions."""
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 33, 38, 52])
    
    # Manual calculation for verification (or use sklearn directly for some)
    # MAE: (|10-12| + |20-18| + |30-33| + |40-38| + |50-52|) / 5
    #    = (2 + 2 + 3 + 2 + 2) / 5 = 11 / 5 = 2.2
    # MSE: (2^2 + (-2)^2 + 3^2 + (-2)^2 + 2^2) / 5
    #    = (4 + 4 + 9 + 4 + 4) / 5 = 25 / 5 = 5.0
    # RMSE: sqrt(5.0) = 2.2360679...
    # R2: Use sklearn.metrics.r2_score
    # MAPE: ((|2|/10) + (|-2|/20) + (|3|/30) + (|-2|/40) + (|2|/52)) / 5 * 100
    #     = (0.2 + 0.1 + 0.1 + 0.05 + 0.03846) / 5 * 100
    #     = (0.48846) / 5 * 100 = 0.097692 * 100 = 9.7692%
        
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt(expected_mse)
    expected_r2 = r2_score(y_true, y_pred)
    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    expected_metrics = {
        'mae': expected_mae,
        'mse': expected_mse,
        'rmse': expected_rmse,
        'r2': expected_r2,
        'mape': expected_mape
    }
    
    actual_metrics = calculate_metrics(y_true, y_pred)
    
    for key in expected_metrics:
        assert np.isclose(actual_metrics[key], expected_metrics[key]), f"Metric {key} failed for imperfect prediction"

def test_calculate_metrics_with_zeros_in_true():
    """Test calculate_metrics when y_true contains zeros (affects MAPE)."""
    y_true = np.array([0, 1, 2, 0, 5])
    y_pred = np.array([0.1, 1.1, 2.1, 0.1, 5.1]) # Small errors
    
    # MAPE calculation in calculate_metrics handles division by zero by replacing inf with a large number.
    # Let's verify other metrics.
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt(expected_mse)
    expected_r2 = r2_score(y_true, y_pred)
    
    # Manual MAPE considering the handling in the source code:
    # errors = np.abs((y_true - y_pred) / y_true)
    # errors will be [inf, 0.1, 0.05, inf, 0.02]
    # np.where(np.isinf(errors), large_value, errors)
    # Assuming large_value is 1e9 (as per source code if not defined)
    # (1e9 + 0.1 + 0.05 + 1e9 + 0.02) / 5 * 100
    # This will be a very large number.
    
    actual_metrics = calculate_metrics(y_true, y_pred)
    
    assert np.isclose(actual_metrics['mae'], expected_mae)
    assert np.isclose(actual_metrics['mse'], expected_mse)
    assert np.isclose(actual_metrics['rmse'], expected_rmse)
    assert np.isclose(actual_metrics['r2'], expected_r2)
    
    # For MAPE, when y_true has zeros, the function replaces inf with a large number.
    # Check if it's a large positive number.
    assert actual_metrics['mape'] > 1e5 # Expect a large value due to division by zero handling

    # Test with y_true being all zeros
    y_true_all_zeros = np.array([0, 0, 0])
    y_pred_some_val = np.array([1, 1, 1])
    actual_metrics_all_zeros = calculate_metrics(y_true_all_zeros, y_pred_some_val)
    assert actual_metrics_all_zeros['mape'] > 1e5 # Should be large

def test_calculate_metrics_zero_variance_true():
    """Test R2 score when y_true has zero variance."""
    y_true = np.array([5, 5, 5, 5, 5])
    
    # Case 1: Predictions are also constant and equal to y_true (perfect prediction)
    y_pred_perfect = np.array([5, 5, 5, 5, 5])
    metrics_perfect = calculate_metrics(y_true, y_pred_perfect)
    # R2 score is 1.0 if both y_true and y_pred are constant and equal.
    # sklearn.metrics.r2_score behavior: if y_true is constant, it's 1.0 if y_pred is also constant
    # and equal to y_true, and 0.0 if y_pred is constant but different. If y_pred is not constant,
    # it can be negative. The function's R2 should align.
    assert np.isclose(metrics_perfect['r2'], 1.0) 
    assert np.isclose(metrics_perfect['mae'], 0.0)

    # Case 2: Predictions are constant but different from y_true
    y_pred_constant_diff = np.array([6, 6, 6, 6, 6])
    metrics_constant_diff = calculate_metrics(y_true, y_pred_constant_diff)
    # R2 score is 0.0 if y_true is constant and y_pred is constant but different.
    assert np.isclose(metrics_constant_diff['r2'], 0.0) 
    assert np.isclose(metrics_constant_diff['mae'], 1.0)

    # Case 3: Predictions are not constant
    y_pred_not_constant = np.array([5, 6, 5, 6, 5])
    metrics_not_constant = calculate_metrics(y_true, y_pred_not_constant)
    # R2 will be negative here. R2 = 1 - (SS_res / SS_tot). If SS_tot is 0, R2 behavior is specific.
    # sklearn's r2_score for constant y_true and non-constant y_pred will result in a score
    # that reflects how much worse the model is than just predicting the mean.
    # If y_true is constant, SS_tot = 0.
    # If y_pred is also constant and y_pred = y_true, then SS_res = 0, R2 = 1.0
    # If y_pred is constant and y_pred != y_true, then SS_res > 0, R2 = 0.0 (by convention in sklearn)
    # If y_pred is not constant, then SS_res > 0.
    # The implementation in baq.core.evaluation.py uses sklearn.metrics.r2_score, so it should match.
    expected_r2_not_constant = r2_score(y_true, y_pred_not_constant)
    assert np.isclose(metrics_not_constant['r2'], expected_r2_not_constant)


def test_calculate_metrics_empty_input():
    """Test calculate_metrics with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])
    
    actual_metrics = calculate_metrics(y_true, y_pred)
    
    # For empty arrays, MAE, MSE, RMSE should be 0 or NaN depending on sklearn version.
    # R2 is typically undefined (NaN or error). MAPE is also problematic.
    # The current implementation of calculate_metrics would likely result in NaNs from np.mean([])
    # or errors from sklearn metrics if they don't handle empty arrays gracefully by returning NaN.
    # Let's assume the function should return NaNs or handle it gracefully.
    # np.mean([]) results in NaN.
    # sklearn metrics on empty arrays often result in nan or specific values.
    # e.g. mean_absolute_error([], []) is 0.0 if warn_ FutureWarning true, else nan
    # r2_score([], []) is nan

    # Based on calculate_metrics implementation:
    # mae, mse, r2 from sklearn. If they return 0 for empty, then rmse will be 0.
    # mape will be np.mean([])*100 = nan * 100 = nan
    
    # If sklearn's metrics return 0 for empty inputs (as some versions might with specific params):
    # assert np.isclose(actual_metrics['mae'], 0.0)
    # assert np.isclose(actual_metrics['mse'], 0.0)
    # assert np.isclose(actual_metrics['rmse'], 0.0)
    # For R2, it's often NaN or an arbitrary value like 0.0 when y_true is empty or constant.
    # If y_true is empty, r2_score returns nan.
    # assert np.isnan(actual_metrics['r2']) 
    # assert np.isnan(actual_metrics['mape'])

    # A safer check for empty inputs, assuming NaNs are the expected outcome for ill-defined metrics:
    # This depends on how robustly the underlying sklearn functions handle empty arrays.
    # If calculate_metrics is expected to raise an error for empty inputs, test for that instead.
    # Given the direct use of sklearn functions, their behavior for empty arrays will dictate output.
    # Let's check for NaNs as it's a common result for undefined metrics on empty sets.
    for key in ['mae', 'mse', 'rmse', 'r2', 'mape']:
        if key in ['mae', 'mse', 'rmse'] and actual_metrics[key] == 0.0: 
            # Some sklearn versions return 0 for MAE/MSE on empty. RMSE would be 0.
            pass
        else:
            assert np.isnan(actual_metrics[key]), f"Metric {key} should be NaN for empty inputs or follow sklearn's behavior"


# To run this test file:
# Ensure you are in the root directory of the project.
# Run: pytest tests/unit/core/test_evaluation.py
# (Or `uv run pytest tests/unit/core/test_evaluation.py` as per user's setup)
