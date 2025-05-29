import pytest
import os
from unittest import mock

# Function to be tested
from baq.steps.save_artifacts import save_artifacts

# Functions from baq.utils.artifacts will be mocked
# import baq.utils.artifacts

@pytest.fixture
def mock_artifact_utils():
    """Mocks all save_* functions from baq.utils.artifacts."""
    with mock.patch('baq.utils.artifacts.save_model') as mock_save_model, \
         mock.patch('baq.utils.artifacts.save_processor') as mock_save_processor, \
         mock.patch('baq.utils.artifacts.save_metrics') as mock_save_metrics, \
         mock.patch('baq.utils.artifacts.save_plots') as mock_save_plots:
        
        yield {
            "save_model": mock_save_model,
            "save_processor": mock_save_processor,
            "save_metrics": mock_save_metrics,
            "save_plots": mock_save_plots
        }

@pytest.fixture
def sample_config_for_saving():
    """Provides a sample configuration dictionary for artifact paths."""
    return {
        "artifacts": {
            "model": {"path": "models_dir", "filename": "model.pkl"},
            "processors": {"path": "procs_dir", "filename": "processor.pkl"},
            "metrics": {"path": "metrics_dir", "filename": "metrics.json"},
            "plots": {"path": "plots_dir"} 
            # No filename for plots as save_plots in utils handles individual names
        }
    }

def test_save_artifacts_calls_utilities_correctly(mock_artifact_utils, sample_config_for_saving):
    mock_model_obj = mock.MagicMock(name="MockModel")
    mock_processor_obj = mock.MagicMock(name="MockProcessor")
    mock_metrics_dict = {"acc": 0.99}
    mock_plots_dict = {"loss": mock.MagicMock(name="LossPlotFigure")}
    
    base_artifacts_path = "/tmp/test_artifacts" # Example base path
    config = sample_config_for_saving

    save_artifacts(
        model=mock_model_obj,
        processor=mock_processor_obj,
        metrics=mock_metrics_dict,
        plots=mock_plots_dict,
        artifacts_path=base_artifacts_path,
        config=config
    )

    # 1. Assert save_model was called correctly
    expected_model_path = os.path.join(base_artifacts_path, 
                                       config["artifacts"]["model"]["path"], 
                                       config["artifacts"]["model"]["filename"])
    mock_artifact_utils["save_model"].assert_called_once_with(mock_model_obj, expected_model_path)

    # 2. Assert save_processor was called correctly
    expected_processor_path = os.path.join(base_artifacts_path,
                                           config["artifacts"]["processors"]["path"],
                                           config["artifacts"]["processors"]["filename"])
    mock_artifact_utils["save_processor"].assert_called_once_with(mock_processor_obj, expected_processor_path)

    # 3. Assert save_metrics was called correctly
    expected_metrics_path = os.path.join(base_artifacts_path,
                                         config["artifacts"]["metrics"]["path"],
                                         config["artifacts"]["metrics"]["filename"])
    mock_artifact_utils["save_metrics"].assert_called_once_with(mock_metrics_dict, expected_metrics_path)

    # 4. Assert save_plots was called correctly
    expected_plots_path = os.path.join(base_artifacts_path, config["artifacts"]["plots"]["path"])
    mock_artifact_utils["save_plots"].assert_called_once_with(mock_plots_dict, expected_plots_path)


# To run: pytest tests/unit/steps/test_save_artifacts.py
# (or `uv run pytest tests/unit/steps/test_save_artifacts.py`)
