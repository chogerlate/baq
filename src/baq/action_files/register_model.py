import os
import sys
import logging
import wandb
import argparse
from typing import Optional, Dict, Any


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


def validate_run_id(api: wandb.Api, run_id: str, logger: logging.Logger) -> Optional[wandb.apis.public.Run]:
    """
    Validate that a run exists and is accessible.
    
    Args:
        api: W&B API object
        run_id: The run ID to validate
        logger: Logger instance
        
    Returns:
        The run object if valid, None otherwise
    """
    try:
        # Try to get the run
        logger.info(f"🔍 Validating run ID: {run_id}")
        
        # Check if run_id contains entity/project
        if '/' in run_id:
            run = api.run(run_id)
        else:
            # Try with default project
            project = os.getenv("WANDB_PROJECT", "baq")
            entity = os.getenv("WANDB_ENTITY", "chogerlate")
            full_run_id = f"{entity}/{project}/{run_id}"
            logger.info(f"Using full run path: {full_run_id}")
            run = api.run(full_run_id)
        
        # Verify run is finished
        if run.state != "finished":
            logger.error(f"❌ Run {run_id} is in state '{run.state}'. Only finished runs can be registered.")
            return None
            
        return run
    
    except Exception as e:
        logger.error(f"❌ Invalid run ID '{run_id}': {str(e)}")
        logger.info("Make sure:")
        logger.info("1. The run ID is correct")
        logger.info("2. You have access to the run")
        logger.info("3. The run exists in the correct project")
        logger.info(f"4. Try using the full path: entity/project/{run_id}")
        return None


def register_model(run_id: str, collection_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Register a model artifact from a run to the W&B model registry.
    
    Args:
        run_id: The W&B run ID containing the model artifact
        collection_path: The model registry collection path (e.g., "entity/project/model-name")
        logger: Logger instance
    
    Returns:
        Dict with registration status and details
    """
    api = wandb.Api()
    
    try:
        # Validate run first
        run = validate_run_id(api, run_id, logger)
        if not run:
            raise ValueError(f"Invalid run ID: {run_id}")
        
        # Get model artifacts from the run
        artifacts = list(run.logged_artifacts())
        model_artifacts = [artifact for artifact in artifacts if artifact.type == "model"]
        
        if not model_artifacts:
            logger.warning(f"⚠️ No model artifacts found in run {run_id}")
            if artifacts:
                logger.info(f"Available artifacts: {[f'{a.name}:{a.type}' for a in artifacts]}")
            else:
                logger.info("No artifacts found in this run")
            raise ValueError("No model artifacts found")
        
        # Use the first model artifact (or could be made configurable)
        model_artifact = model_artifacts[0]
        logger.info(f"📦 Found model artifact: {model_artifact.name}:{model_artifact.version}")
        
        # Link the artifact to the model collection
        logger.info(f"🔗 Linking to collection: {collection_path}")
        linked_artifact = model_artifact.link(collection_path)
        
        result = {
            "status": "success",
            "run_id": run_id,
            "model_name": model_artifact.name,
            "model_version": model_artifact.version,
            "collection": collection_path
        }
        
        logger.info(f"✅ Successfully registered model {model_artifact.name}:{model_artifact.version} to {collection_path}")
        return result
        
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
        result = register_model(args.run_id, args.collection_path, logger)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
