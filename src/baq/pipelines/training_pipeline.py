"""
This module implements the complete training pipeline for forecasting models.
- Orchestrates the entire model training workflow from data loading to evaluation
- Integrates various steps including data processing, feature engineering, and model training
- Supports configurable pipeline parameters for flexibility across different use cases
- Implements logging and error handling for robust pipeline execution
- Manages model artifacts and metadata for tracking experiments
- Provides interfaces for monitoring training progress and results
- Ensures reproducibility of the training process through consistent workflows
"""
import os
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb
from dotenv import load_dotenv
import hydra

from baq.utils.artifacts import create_artifact_directories
from baq.steps.load_data import load_data
from baq.steps.process import process_train_data
from baq.steps.train import train_model
from baq.steps.evaluate import evaluate_model
from baq.steps.save_artifacts import save_artifacts
from baq.steps.monitoring_report import MonitoringReport
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_wandb(config: DictConfig):
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        config: Configuration dictionary
    """
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    return wandb.init(
        name=f'{config["model"]["model_type"]}-{wandb.util.generate_id()}',
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        job_type=config["wandb"]["job_type"],
        tags=config["wandb"]["tags"] + [config["model"]["model_type"]]
    )



def training_pipeline(config: DictConfig) -> None:
    """
    Main training pipeline.
    
    Args:
        config: Hydra configuration
    """
    logger.info(f"Using configuration: \n{OmegaConf.to_yaml(config)}")
    
    # Get model type
    model_type = config["model"]["model_type"]
    # Create artifact directories
    artifact_path = create_artifact_directories(config=config)
    
    # Load and preprocess data
    df = load_data(config["data"]["raw_data_path"])
    
    # Create features and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, processor = process_train_data(
        df,
        target_column=config["training"]["target_column"],
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    
    # Train model
    model, avg_metrics = train_model(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        model_name=model_type,
        model_params=config["model"][model_type]["model_params"],
        model_training_params=config["model"][model_type]["training_params"],
        training_config=config["training"]
    )
    
    # Evaluate model
    single_step_metrics, multi_step_metrics, plots = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        forecast_horizon=config["training"]["forecast_horizon"],
        sequence_length=config["training"]["sequence_length"]
    )
    

    metrics = {
        "train_metrics": avg_metrics,
        "single_step_metrics": single_step_metrics,
        "multi_step_metrics": multi_step_metrics
    }

    # Create monitoring report
    monitoring_report = MonitoringReport(
        model=model,
        train_data=X_train,
        test_data=X_test,
        config=config
    )
    report = monitoring_report.create_monitoring_report()
    monitoring_report.save_monitoring_report(report, os.path.join(artifact_path, config["artifacts"]["reports"]["path"], config["artifacts"]["reports"]["filename"]))

    # Save artifacts
    save_artifacts(
        model=model,
        processor=processor,
        metrics=metrics,
        plots=plots,
        artifacts_path=artifact_path,
        config=config,
    )
    
    
    print("\n========== Training completed ==========")
    logger.info(f"Best model: {model_type.upper()}")
    logger.info(f"Metrics: {avg_metrics}")
    logger.info(f"Artifacts saved to {artifact_path}")
    print("========================================\n")

    # Section: Logging
    if config["experiment_tracking_status"]:
        run = setup_wandb(config=config)
        
        # Artifacts
        model_artifact = wandb.Artifact(
            name=f"{config['model']['model_type']}_model",
            type="model",
            description=f"""
            Model for {config['training']['target_column']} forecasting. 
            Evaluated on {config['training']['forecast_horizon']}-step ahead predictions 
            and {config['training']['n_splits']}-fold cross-validation.
            The Results are based on the following metrics:
            - MAE: {metrics['single_step_metrics']['mae']}
            - MAPE: {metrics['single_step_metrics']['mape']}
            - MSE: {metrics['single_step_metrics']['mse']}
            - RMSE: {metrics['single_step_metrics']['rmse']}
            - R2: {metrics['single_step_metrics']['r2']}
            """
        )


        processor_artifact = wandb.Artifact(
            name=f"{config['model']['model_type']}_processor",
            type="processor",
            description=f"Processor for {config['training']['target_column']} forecasting."
        )

        run_log_artifact = wandb.Artifact(
            name=f"{run.id}_run_log",
            type="run_log",
            description=f"Run log for {run.id}"
        )

        monitoring_report_artifact = wandb.Artifact(
            name=f"{run.id}_monitoring_report",
            type="monitoring_report",
            description=f"Monitoring report for {run.id}"
        )

        monitoring_report_artifact.add_file(os.path.join(artifact_path, config["artifacts"]["reports"]["path"], config["artifacts"]["reports"]["filename"]), "monitoring_report")
        run_log_artifact.add_file(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'run.log'), "run_log")
        processor_artifact.add_file(os.path.join(artifact_path, config["artifacts"]["processors"]["path"], config["artifacts"]["processors"]["filename"]), "processor")
        model_artifact.add_file(os.path.join(artifact_path, config["artifacts"]["model"]["path"], config["artifacts"]["model"]["filename"]), "model")
        


        # Evaluation results
        log_plots = {f"{plot_name}": wandb.Image(plot) for plot_name, plot in plots.items()}
        evaluation_results = {
            **metrics,
            **log_plots
        }

        run.log(evaluation_results)
        run.log_artifact(processor_artifact)
        run.log_artifact(model_artifact)
        run.log_artifact(run_log_artifact)
        run.log_artifact(monitoring_report_artifact)

        # Register the model
        if config["wandb"]["register_model"]:
            target_path = os.getenv("WANDB_MODEL_REGISTRY_COLLECTION_PATH")
            if target_path is not None:
                run.link_artifact(
                    artifact=model_artifact,
                    target_path=target_path
                )
                logger.info(f"Model registered to {target_path}")
            else:
                logger.warning("WANDB_MODEL_REGISTRY_COLLECTION_PATH environment variable is not set. Model registration skipped.")
        
        # Finish the run
        run.finish()
        