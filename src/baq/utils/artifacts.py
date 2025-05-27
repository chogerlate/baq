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