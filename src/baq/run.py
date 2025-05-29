"""
Main Entry Point for Bangkok Air Quality Forecasting Pipeline.

This module serves as the primary entry point for running the Bangkok Air Quality
forecasting training pipeline. It handles configuration management, logging setup,
and pipeline orchestration using Hydra for configuration management.

Features:
- Hydra-based configuration management with YAML configs
- Comprehensive logging setup with structured output
- Environment variable loading for sensitive configurations
- Main training pipeline orchestration
- Error handling and graceful execution flow

The module coordinates the complete ML pipeline execution:
1. Configuration loading and resolution via Hydra
2. Environment setup and logging initialization
3. Training pipeline execution with all steps
4. Graceful error handling and cleanup

Example:
    Run the training pipeline with default configuration:
    >>> python src/baq/run.py
    
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from baq.pipelines.training_pipeline import training_pipeline
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

@hydra.main(config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training pipeline.
    
    Args:
        config: Hydra configuration
    """
    # Resolve the config
    config = OmegaConf.to_container(config, resolve=True)    
    training_pipeline(config)

if __name__ == "__main__":
    main() 