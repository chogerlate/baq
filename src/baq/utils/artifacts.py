"""
Artifact Management Utilities for Bangkok Air Quality Forecasting.

This module provides comprehensive utilities for managing model artifacts including
saving, loading, and organizing various types of artifacts generated during the
machine learning pipeline.

Features:
- Model serialization and persistence (supports multiple formats)
- Time series preprocessor saving and loading
- Metrics storage and retrieval in JSON format
- Plot visualization management
- Report generation and storage
- Artifact directory structure creation and management
- Support for various model types (LSTM, Random Forest, XGBoost)

The module handles the complete artifact lifecycle:
1. Model weights and architecture persistence
2. Data preprocessing pipeline serialization
3. Performance metrics documentation
4. Visualization plots storage
5. Training reports and summaries
6. Directory structure organization for reproducibility

Example:
    >>> # Save model artifacts
    >>> save_model(trained_model, "models/lstm_model.pkl")
    >>> save_processor(data_processor, "processors/processor.pkl")
    >>> save_metrics(evaluation_metrics, "metrics/results.json")
    >>> save_plots(visualization_plots, "plots/")
    >>> 
    >>> # Create organized directory structure
    >>> artifacts_path = create_artifact_directories(config)
"""


import joblib
import json
import os

def save_model(model, model_path):
    joblib.dump(model, model_path)

def save_processor(processor, processor_path):
    """
    Save the time series processor object.
    
    Args:
        processor: The processor object to save
        processor_path: Path to save the processor
    """
    joblib.dump(processor, processor_path)

def save_metrics(metrics, metrics_path):
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

def save_plots(plots, plots_path):
    for plot_name, plot in plots.items():
        plot.savefig(os.path.join(plots_path, f"{plot_name}.png"))

def save_reports(reports, reports_path):
    for report_name, report in reports.items():
        with open(os.path.join(reports_path, f"{report_name}.md"), "w") as f:
            f.write(report)

def create_artifact_directories(
        config: dict,
):
    artifacts_path = os.path.abspath(config["artifacts"]["base_path"])
    os.makedirs(artifacts_path, exist_ok=True)
    os.makedirs(os.path.join(artifacts_path, config["artifacts"]["model"]["path"]), exist_ok=True)
    os.makedirs(os.path.join(artifacts_path, config["artifacts"]["metrics"]["path"]), exist_ok=True)
    os.makedirs(os.path.join(artifacts_path, config["artifacts"]["plots"]["path"]), exist_ok=True)
    os.makedirs(os.path.join(artifacts_path, config["artifacts"]["reports"]["path"]), exist_ok=True)
    os.makedirs(os.path.join(artifacts_path, config["artifacts"]["processors"]["path"]), exist_ok=True)
    return artifacts_path