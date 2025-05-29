"""
Artifact Saving Module for Bangkok Air Quality Forecasting.

This module provides functionality to save trained models, preprocessors, evaluation metrics,
and visualization plots to specified artifact directories. It supports various model types
including LSTM, Random Forest, and XGBoost models, along with their associated artifacts.

Features:
- Model serialization and saving (supports multiple formats)
- Time series preprocessor persistence
- Metrics storage in JSON format
- Plot visualization saving
- Configurable artifact directory structure
- Organized artifact management for reproducibility

The module handles the complete artifact saving pipeline:
1. Model weights and architecture persistence
2. Data preprocessing pipeline serialization
3. Performance metrics documentation
4. Visualization plots storage

Example:
    >>> save_artifacts(
    ...     model=trained_model,
    ...     processor=data_processor,
    ...     metrics=evaluation_metrics,
    ...     plots=visualization_plots,
    ...     artifacts_path="./artifacts",
    ...     config=training_config
    ... )
"""
import os
from baq.utils.artifacts import save_model, save_metrics, save_plots, save_processor

def save_artifacts(
    model: object,
    processor: object,
    metrics: dict,
    plots: dict,
    artifacts_path: str,
    config: dict,
) -> None:
    """
    Save the model, processor, metrics and plots to the artifacts path.
    
    Args:
        model: Trained model object
        processor: Time series preprocessor 
        metrics: Dictionary of metrics
        plots: Dictionary of plots
        artifacts_path: Path to save artifacts
        config: Configuration dictionary
    """
    # Prepare paths
    model_path = os.path.join(artifacts_path, config["artifacts"]["model"]["path"], config["artifacts"]["model"]["filename"])
    metrics_path = os.path.join(artifacts_path, config["artifacts"]["metrics"]["path"], config["artifacts"]["metrics"]["filename"])
    plots_path = os.path.join(artifacts_path, config["artifacts"]["plots"]["path"])
    processor_path = os.path.join(artifacts_path, config["artifacts"]["processors"]["path"], 
                                 config["artifacts"]["processors"]["filename"])
    
    # Save artifacts
    save_model(model, model_path)
    save_processor(processor, processor_path)
    save_metrics(metrics, metrics_path)
    save_plots(plots, plots_path)