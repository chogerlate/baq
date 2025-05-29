"""
Training Pipeline Module for Bangkok Air Quality Forecasting.

This module implements the complete training pipeline for PM2.5 forecasting models.
It orchestrates the entire workflow from data loading to model evaluation and artifact
management.

Features:
- End-to-end training workflow
- Experiment tracking with W&B
- Model artifact management
- Performance monitoring
- Reproducible training process

The pipeline follows these steps:
1. Data loading and validation
2. Feature processing and engineering
3. Model training and validation
4. Performance evaluation
5. Artifact saving and logging
6. Monitoring report generation

Example:
    >>> config = OmegaConf.load('configs/training.yaml')
    >>> training_pipeline(config)
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_wandb(config: DictConfig) -> object:
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        config: Configuration dictionary containing W&B settings
    
    Returns:
        Initialized W&B run object
    
    Note:
        Requires WANDB_API_KEY environment variable
    """
    logger.info("Setting up Weights & Biases...")
    
    # Validate W&B API key
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY environment variable not set")
    
    # Initialize W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        name=f'{config["model"]["model_type"]}-{wandb.util.generate_id()}',
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        job_type=config["wandb"]["job_type"],
        tags=config["wandb"]["tags"] + [config["model"]["model_type"]]
    )
    
    logger.info(f"W&B run initialized: {run.name}")
    return run

def create_wandb_artifacts(
    config: DictConfig,
    metrics: Dict[str, Any],
    run: object
) -> None:
    """
    Create and log W&B artifacts for the training run.
    
    Args:
        config: Training configuration
        metrics: Model performance metrics
        run: Active W&B run
    """
    logger.info("Creating W&B artifacts...")
    
    # Model artifact
    model_artifact = wandb.Artifact(
        name=f"{config['model']['model_type']}_model",
        type="model",
        description=f"""
        Model for {config['training']['target_column']} forecasting. 
        Evaluated on {config['training']['forecast_horizon']}-step ahead predictions 
        and {config['training']['n_splits']}-fold cross-validation.
        Results:
        - MAE: {metrics['single_step_metrics']['mae']:.4f}
        - MAPE: {metrics['single_step_metrics']['mape']:.4f}
        - MSE: {metrics['single_step_metrics']['mse']:.4f}
        - RMSE: {metrics['single_step_metrics']['rmse']:.4f}
        - R2: {metrics['single_step_metrics']['r2']:.4f}
        """
    )
    
    # Processor artifact
    processor_artifact = wandb.Artifact(
        name=f"{config['model']['model_type']}_processor",
        type="processor",
        description=f"Processor for {config['training']['target_column']} forecasting."
    )
    
    # Run log artifact
    run_log_artifact = wandb.Artifact(
        name=f"{run.id}_run_log",
        type="run_log",
        description=f"Run log for {run.id}"
    )
    
    # Monitoring report artifact
    monitoring_report_artifact = wandb.Artifact(
        name=f"{run.id}_monitoring_report",
        type="monitoring_report",
        description=f"Monitoring report for {run.id}"
    )
    
    # Log artifacts
    run.log_artifact(model_artifact)
    run.log_artifact(processor_artifact)
    run.log_artifact(run_log_artifact)
    run.log_artifact(monitoring_report_artifact)
    
    logger.info("W&B artifacts created and logged")

def training_pipeline(config: DictConfig) -> None:
    """
    Execute the complete training pipeline.
    
    Args:
        config: Hydra configuration object containing all settings
    
    The pipeline performs these steps:
    1. Setup and validation
    2. Data loading and processing
    3. Model training
    4. Evaluation
    5. Artifact management
    6. Monitoring
    
    Note:
        Requires proper configuration in config/training.yaml
    """
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Get model configuration
    model_type = config["model"]["model_type"]
    logger.info(f"Training model type: {model_type}")
    
    # Create artifact directories
    artifact_path = create_artifact_directories(config=config)
    logger.info(f"Artifact directory created: {artifact_path}")
    
    # Load and preprocess data
    logger.info("Loading raw data...")
    df = load_data(config["data"]["raw_data_path"])
    
    logger.info("Processing training data...")
    X_train, y_train, X_val, y_val, X_test, y_test, processor = process_train_data(
        df,
        target_column=config["training"]["target_column"],
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    logger.info(f"Data split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    logger.info("Training model...")
    model, avg_metrics = train_model(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        model_name=model_type,
        model_params=config["model"][model_type]["model_params"],
        model_training_params=config["model"][model_type]["training_params"],
        training_config=config["training"]
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    single_step_metrics, multi_step_metrics, plots = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        forecast_horizon=config["training"]["forecast_horizon"],
        sequence_length=config["training"]["sequence_length"]
    )
    
    # Combine metrics
    metrics = {
        "train_metrics": avg_metrics,
        "single_step_metrics": single_step_metrics,
        "multi_step_metrics": multi_step_metrics
    }
    
    # Create monitoring report
    logger.info("Generating monitoring report...")
    monitoring_report = MonitoringReport(
        model=model,
        train_data=X_train,
        test_data=X_test,
        config=config
    )
    report = monitoring_report.create_monitoring_report()
    report_path = os.path.join(
        artifact_path,
        config["artifacts"]["reports"]["path"],
        config["artifacts"]["reports"]["filename"]
    )
    monitoring_report.save_monitoring_report(report, report_path)
    
    # Save artifacts
    logger.info("Saving model artifacts...")
    save_artifacts(
        model=model,
        processor=processor,
        metrics=metrics,
        plots=plots,
        artifacts_path=artifact_path,
        config=config,
    )
    
    # Print training summary
    print("\n========== Training Summary ==========")
    logger.info(f"Model: {model_type.upper()}")
    logger.info(f"Metrics: {avg_metrics}")
    logger.info(f"Artifacts saved to: {artifact_path}")
    print("=====================================\n")
    
    # Handle experiment tracking
    if config["experiment_tracking_status"]:
        logger.info("Setting up experiment tracking...")
        run = setup_wandb(config=config)
        create_wandb_artifacts(config, metrics, run)
        logger.info("Experiment tracking completed")
    
    logger.info("Training pipeline completed successfully")
    