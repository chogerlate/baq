import os
import wandb
import argparse
import logging
import sys


def setup_logging() -> logging.Logger:
    """Set up logging for GitHub Actions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def register_model(run_id: str, collection_path: str, logger: logging.Logger) -> wandb.Artifact:
    """
    Register a model artifact from a run to the W&B model registry.
    
    Args:
        run_id: The W&B run ID containing the model artifact
        collection_path: The model registry collection path (e.g., "entity/project/model-name")
        logger: Logger instance for output
    
    Returns:
        The linked model artifact or None if no model artifacts found
    """
    api = wandb.Api()
    
    try:
        # Get the run
        logger.info(f"🔍 Fetching run: {run_id}")
        run = api.run(run_id)
        
        # Get model artifacts from the run
        artifacts = list(run.logged_artifacts())
        model_artifacts = [artifact for artifact in artifacts if artifact.type == "model"]
        
        if not model_artifacts:
            logger.warning(f"⚠️ No model artifacts found in run {run_id}")
            logger.info(f"Available artifacts: {[f'{a.name}:{a.type}' for a in artifacts]}")
            return None
        
        # Use the first model artifact (or could be made configurable)
        model_artifact = model_artifacts[0]
        logger.info(f"📦 Found model artifact: {model_artifact.name}:{model_artifact.version}")
        
        # Link the artifact to the model collection
        logger.info(f"🔗 Linking to collection: {collection_path}")
        linked_artifact = model_artifact.link(collection_path)
        
        logger.info(f"✅ Successfully registered model {model_artifact.name}:{model_artifact.version} to {collection_path}")
        return linked_artifact
        
    except Exception as e:
        logger.error(f"❌ Error registering model: {str(e)}")
        raise


def main():
    """Main function to handle model registration."""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Register a model in the W&B model registry")
    parser.add_argument("--run-id", type=str, required=True, help="The run ID of the model to register")
    parser.add_argument("--collection-path", 
                       default=os.environ.get("WANDB_MODEL_REGISTRY_COLLECTION_PATH", "chogerlate/wandb-registry-model/baq-forecastors"), 
                       help="W&B registry collection path")
    args = parser.parse_args()
    
    try:
        register_model(args.run_id, args.collection_path, logger)
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
