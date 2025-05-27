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