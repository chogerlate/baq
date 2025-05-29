import os
import pytest
from unittest import mock
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import tempfile
import shutil

# Import the training_pipeline function
# This will work because of tests/context.py, but the linter might not see it.
from baq.pipelines.training_pipeline import training_pipeline 

# Minimal configuration for the smoke test
# Adjust paths and parameters as needed for a quick run
MINIMAL_CONFIG = {
    'experiment_tracking_status': False,  # Disable W&B for smoke test
    'data': {
        'raw_data_path': 'smoke_test_raw_data.csv', # Path for temporary smoke test data
    },
    'model': {
        'model_type': 'random_forest', # Use a simple model
        'random_forest': {
            'model_params': {
                'n_estimators': 1, # Minimal parameters
                'max_depth': 1,
            },
            'training_params': {}, # No specific training params needed for RF smoke
        },
        # Minimal configs for other models if your pipeline tries to load them
        'xgboost': {
            'model_params': {'n_estimators': 1, 'max_depth': 1},
            'training_params': {},
        },
        'lstm': {
            'model_params': {'n_layers': 1, 'hidden_size': 8, 'dropout': 0.1},
            'training_params': {'epochs': 1, 'batch_size': 1},
        },
    },
    'artifacts': {
        'base_path': './smoke_test_artifacts', # Use a temporary artifacts directory
        'model': {'path': 'models', 'filename': 'smoke_model.joblib'},
        'processors': {'path': 'processors', 'filename': 'smoke_processor.joblib'},
        'metrics': {'path': 'metrics', 'filename': 'smoke_metrics.json'},
        'plots': {'path': 'plots'},
        'reports': {'path': 'reports', 'filename': 'smoke_monitoring_report.html'},
    },
    'training': {
        'forecast_horizon': 1,
        'sequence_length': 1, # Minimal sequence length
        'target_column': 'pm2_5', # Ensure this matches dummy data
        'n_splits': 2, # Minimal splits
        'test_size': 0.5, # Use a small test size for speed
        'random_state': 42,
    },
    # Add wandb section if your pipeline strictly requires it, even if status is false
    'wandb': {
      'tags': ["smoke_test"],
      'log_model': False,
      'register_model': False,
      'job_type': "smoke_test_training"
    }
}

@pytest.fixture(scope="module")
def smoke_test_environment():
    # Create a temporary directory for artifacts and data
    temp_dir = tempfile.mkdtemp()
    artifacts_path = os.path.join(temp_dir, "smoke_test_artifacts")
    raw_data_path = os.path.join(temp_dir, MINIMAL_CONFIG['data']['raw_data_path'])

    # Create dummy raw data CSV for the smoke test
    # Adjust columns to match what your pipeline expects, including the target_column
    dummy_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', 
                                    '2023-01-01 02:00:00', '2023-01-01 03:00:00',
                                    '2023-01-01 04:00:00', '2023-01-01 05:00:00',
                                    '2023-01-01 06:00:00', '2023-01-01 07:00:00']),
        'pm2_5': [10, 12, 15, 13, 16, 18, 20, 22], # Target column
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [8, 7, 6, 5, 4, 3, 2, 1]
    })
    dummy_data.to_csv(raw_data_path, index=False)

    # Update config to use the temporary paths
    test_config = OmegaConf.create(MINIMAL_CONFIG)
    test_config.data.raw_data_path = raw_data_path
    test_config.artifacts.base_path = artifacts_path
    
    # Ensure artifact subdirectories are correctly referenced if create_artifact_directories needs them
    # For example, if it expects artifacts.model.path to be relative to artifacts.base_path
    # This structure seems to be implied by the config, so it should be fine.

    yield test_config # Provide the config to the test

    # Teardown: remove the temporary directory
    shutil.rmtree(temp_dir)

# Mock environment variables if your pipeline uses them directly (e.g., os.getenv)
# and they are not related to W&B (which is disabled by experiment_tracking_status: False)
@mock.patch.dict(os.environ, {
    "WANDB_API_KEY": "test_key", # Mocked, but W&B is disabled
    "WANDB_PROJECT": "test_project",
    "WANDB_ENTITY": "test_entity",
    # Add any other critical env vars your pipeline might need
})
@mock.patch('baq.pipelines.training_pipeline.setup_wandb') # Mock W&B setup
@mock.patch('baq.pipelines.training_pipeline.create_wandb_artifacts') # Mock W&B artifact creation
def test_training_pipeline_smoke(mock_create_artifacts, mock_setup_wandb, smoke_test_environment):
    """
    Smoke test for the main training_pipeline.
    It runs the pipeline with a minimal configuration to ensure it completes
    and creates expected artifacts.
    """
    test_config = smoke_test_environment

    # Run the training pipeline
    # try-except block to catch issues and provide more context if needed
    try:
        training_pipeline(config=test_config)
    except Exception as e:
        pytest.fail(f"training_pipeline failed with exception: {e}")

    # Assert that W&B setup was not called since experiment_tracking_status is False
    mock_setup_wandb.assert_not_called()
    mock_create_artifacts.assert_not_called()

    # Assert that expected artifact files were created
    base_artifact_path = test_config.artifacts.base_path
    
    model_path = os.path.join(base_artifact_path, 
                              test_config.artifacts.model.path, 
                              test_config.artifacts.model.filename)
    assert os.path.exists(model_path), f"Model artifact not found at {model_path}"

    processor_path = os.path.join(base_artifact_path,
                                  test_config.artifacts.processors.path,
                                  test_config.artifacts.processors.filename)
    assert os.path.exists(processor_path), f"Processor artifact not found at {processor_path}"

    metrics_path = os.path.join(base_artifact_path,
                                test_config.artifacts.metrics.path,
                                test_config.artifacts.metrics.filename)
    assert os.path.exists(metrics_path), f"Metrics artifact not found at {metrics_path}"
    
    report_path = os.path.join(base_artifact_path,
                               test_config.artifacts.reports.path,
                               test_config.artifacts.reports.filename)
    assert os.path.exists(report_path), f"Monitoring report not found at {report_path}"

    # Optionally, check for plot directory (plots are often numerous and names vary)
    plot_dir_path = os.path.join(base_artifact_path, test_config.artifacts.plots.path)
    assert os.path.isdir(plot_dir_path), f"Plot directory not found at {plot_dir_path}"
    assert len(os.listdir(plot_dir_path)) > 0, "Plot directory is empty"
