import os
import sys
import logging
import wandb
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

def promote_model(
    run_id: str,
    target: str = "staging",
    entity: str = "chogerlate", 
    project: str = "wandb-registry-model",
    model_name: str = "baq-forecastors",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Promote a model to a target environment using Weights & Biases.
    
    Args:
        run_id (str): The W&B run ID containing the model to promote
        target (str): Target environment (e.g., "staging", "production")
        entity (str): W&B entity/username
        project (str): W&B project name
        model_name (str): Name of the model to promote
        api_key (str, optional): W&B API key. If None, will use WANDB_API_KEY env var
        
    Returns:
        Dict[str, Any]: Promotion result with status and details
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If promotion fails
    """
    logger = setup_logging()
    
    try:
        # Validate inputs
        if not run_id:
            raise ValueError("run_id cannot be empty")
        if not target:
            raise ValueError("target cannot be empty")
            
        # Initialize W&B API
        api_key_to_use = api_key or os.getenv('WANDB_API_KEY')
        if not api_key_to_use:
            raise ValueError("W&B API key must be provided via parameter or WANDB_API_KEY environment variable")
            
        api = wandb.Api(api_key=api_key_to_use)
        logger.info(f"Initialized W&B API for entity: {entity}")
        
        # Get the run
        run_path = f"{entity}/{project}/{run_id}"
        logger.info(f"Fetching run: {run_path}")
        
        try:
            run = api.run(run_path)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch run {run_path}: {str(e)}")
        
        # Find model artifacts in the run
        model_artifacts = []
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and model_name in artifact.name:
                model_artifacts.append(artifact)
        
        if not model_artifacts:
            raise RuntimeError(f"No model artifacts found with name containing '{model_name}' in run {run_id}")
        
        # Use the latest model artifact
        model_artifact = model_artifacts[-1]  # Get the most recent one
        logger.info(f"Found model artifact: {model_artifact.name} (version: {model_artifact.version})")
        
        # Create alias for promotion
        alias = f"{target}-latest"
        
        # Add the alias to promote the model
        try:
            model_artifact.aliases.append(alias)
            model_artifact.save()
            logger.info(f"Successfully promoted model {model_artifact.name} to {target}")
            
            # Also create a version-specific alias
            version_alias = f"{target}-v{model_artifact.version}"
            model_artifact.aliases.append(version_alias)
            model_artifact.save()
            logger.info(f"Added version-specific alias: {version_alias}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to promote model: {str(e)}")
        
        # Prepare result
        result = {
            "status": "success",
            "run_id": run_id,
            "model_name": model_artifact.name,
            "model_version": model_artifact.version,
            "target_environment": target,
            "aliases": [alias, version_alias],
            "artifact_path": f"{entity}/{project}/{model_artifact.name}:{model_artifact.version}",
            "promotion_time": model_artifact.updated_at.isoformat() if model_artifact.updated_at else None
        }
        
        logger.info("Model promotion completed successfully")
        logger.info(f"Result: {result}")
        
        return result
        
    except Exception as e:
        error_msg = f"Model promotion failed: {str(e)}"
        logger.error(error_msg)
        
        # Return error result for GitHub Actions handling
        return {
            "status": "error",
            "error": error_msg,
            "run_id": run_id,
            "target_environment": target
        }

def main():
    """Main function for command-line usage in GitHub Actions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Promote ML model using W&B")
    parser.add_argument("--run-id", required=True, help="W&B run ID")
    parser.add_argument("--target", default="staging", help="Target environment")
    parser.add_argument("--entity", default="chogerlate", help="W&B entity")
    parser.add_argument("--project", default="wandb-registry-model", help="W&B project")
    parser.add_argument("--model-name", default="baq-forecastors", help="Model name")
    
    args = parser.parse_args()
    
    result = promote_model(
        run_id=args.run_id,
        target=args.target,
        entity=args.entity,
        project=args.project,
        model_name=args.model_name
    )
    
    if result["status"] == "error":
        print(f"::error::{result['error']}")
        sys.exit(1)
    else:
        print(f"::notice::Model promoted successfully to {result['target_environment']}")
        # Set GitHub Actions outputs
        print(f"::set-output name=model_version::{result['model_version']}")
        print(f"::set-output name=artifact_path::{result['artifact_path']}")

if __name__ == "__main__":
    main()