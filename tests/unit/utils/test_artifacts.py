import pytest
import os
import json
from unittest import mock
import tempfile
import shutil
import matplotlib.pyplot as plt # For spec in mock plot

# Functions to be tested
from baq.utils.artifacts import (
    save_model,
    save_processor,
    save_metrics,
    save_plots,
    save_reports,
    create_artifact_directories
)

@pytest.fixture
def temp_dir_fixture():
    """Create a temporary directory for tests that write files/dirs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir) # Clean up

# --- Tests for create_artifact_directories ---
def test_create_artifact_directories(temp_dir_fixture):
    base_path = temp_dir_fixture
    config = {
        "artifacts": {
            "base_path": base_path,
            "model": {"path": "my_models"},
            "metrics": {"path": "my_metrics"},
            "plots": {"path": "my_plots"},
            "reports": {"path": "my_reports"},
            "processors": {"path": "my_processors"}
        }
    }

    returned_path = create_artifact_directories(config)
    assert os.path.abspath(base_path) == returned_path

    # Check if directories were created
    assert os.path.isdir(os.path.join(base_path, "my_models"))
    assert os.path.isdir(os.path.join(base_path, "my_metrics"))
    assert os.path.isdir(os.path.join(base_path, "my_plots"))
    assert os.path.isdir(os.path.join(base_path, "my_reports"))
    assert os.path.isdir(os.path.join(base_path, "my_processors"))

    # Test idempotency (calling again should not fail)
    create_artifact_directories(config)
    assert os.path.isdir(os.path.join(base_path, "my_models"))


# --- Tests for individual save functions ---

@mock.patch('joblib.dump')
def test_save_model(mock_joblib_dump, temp_dir_fixture):
    mock_model_obj = mock.MagicMock()
    model_filename = "test_model.joblib"
    model_path = os.path.join(temp_dir_fixture, model_filename)
    
    save_model(mock_model_obj, model_path)
    
    mock_joblib_dump.assert_called_once_with(mock_model_obj, model_path)

@mock.patch('joblib.dump')
def test_save_processor(mock_joblib_dump, temp_dir_fixture):
    mock_processor_obj = mock.MagicMock()
    processor_filename = "test_processor.joblib"
    processor_path = os.path.join(temp_dir_fixture, processor_filename)
    
    save_processor(mock_processor_obj, processor_path)
    
    mock_joblib_dump.assert_called_once_with(mock_processor_obj, processor_path)

def test_save_metrics(temp_dir_fixture):
    metrics_data = {"mae": 0.5, "mse": 0.025}
    metrics_filename = "test_metrics.json"
    metrics_path = os.path.join(temp_dir_fixture, metrics_filename)
    
    save_metrics(metrics_data, metrics_path)
    
    assert os.path.exists(metrics_path)
    with open(metrics_path, 'r') as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics_data

    # Alternative using mock_open (if not wanting to write actual file)
    # m = mock.mock_open()
    # with mock.patch('builtins.open', m):
    #     with mock.patch('json.dump') as mock_json_dump:
    #         save_metrics(metrics_data, "dummy/path/metrics.json")
    #         m.assert_called_once_with("dummy/path/metrics.json", "w")
    #         mock_json_dump.assert_called_once_with(metrics_data, m())


def test_save_plots(temp_dir_fixture):
    mock_plot1 = mock.MagicMock(spec=plt.Figure)
    mock_plot2 = mock.MagicMock(spec=plt.Figure)
    plots_dict = {
        "accuracy_plot": mock_plot1,
        "loss_plot": mock_plot2
    }
    plots_dir = os.path.join(temp_dir_fixture, "plots_output")
    os.makedirs(plots_dir) # save_plots expects the directory to exist

    save_plots(plots_dict, plots_dir)

    mock_plot1.savefig.assert_called_once_with(os.path.join(plots_dir, "accuracy_plot.png"))
    mock_plot2.savefig.assert_called_once_with(os.path.join(plots_dir, "loss_plot.png"))


def test_save_reports(temp_dir_fixture):
    reports_data = {"summary": "# Title\nDetails...", "errors": "No errors."}
    reports_dir = os.path.join(temp_dir_fixture, "reports_output")
    os.makedirs(reports_dir) # save_reports expects the directory to exist

    save_reports(reports_data, reports_dir)

    report1_path = os.path.join(reports_dir, "summary.md")
    report2_path = os.path.join(reports_dir, "errors.md")

    assert os.path.exists(report1_path)
    with open(report1_path, 'r') as f:
        assert f.read() == "# Title\nDetails..."
    
    assert os.path.exists(report2_path)
    with open(report2_path, 'r') as f:
        assert f.read() == "No errors."

# To run: pytest tests/unit/utils/test_artifacts.py
# (or `uv run pytest tests/unit/utils/test_artifacts.py`)
