import os
import sys
import logging
from wandb import Api
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
    version: str, 
    target: str, 
    collection_path: str,
) -> Dict[str, Any]:
    """
    Promote a model to a target environment using Weights & Biases.
    
    Args:
        version (str): The version of the model to promote
        target (str): Target environment (e.g., "staging", "production") 
        collection_path (str): The collection path of the model to promote
        
    Returns:
        Dict[str, Any]: Promotion result with status and details
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If promotion fails
    """
    if not version or not target or not collection_path:
        raise ValueError("All parameters (version, target, collection_path) are required")
    
    api = Api()
    
    try:
        # Get the model artifact by version
        artifact = api.artifact(f"{collection_path}:{version}")
        
        # Add the target alias (this promotes the model)
        artifact.aliases.append(target)
        artifact.save()
        
        return {
            "status": "success",
            "model": collection_path,
            "version": version,
            "promoted_to": target,
            "artifact_name": artifact.name
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to promote model: {str(e)}")

def main():
    """Main function for command-line usage in GitHub Actions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Promote ML model using W&B")
    parser.add_argument("--version", default=None, help="Version of the model to promote")
    parser.add_argument("--target", default="staging", help="Target environment")
    parser.add_argument("--collection-path", default=os.getenv("WANDB_MODEL_REGISTRY_COLLECTION_PATH", "chogerlate/wandb-registry-model/baq-forecastors"), help="Collection path")
    args = parser.parse_args()
    
    logger = setup_logging()
    try:
        logger.info(f"Starting model promotion: version={args.version}, target={args.target}, collection_path={args.collection_path}")
        result = promote_model(
            version=args.version,
            target=args.target,
            collection_path=args.collection_path,
        )
        logger.info(f"Model promotion successful: {result}")
    except Exception as e:
        logger.error(f"Failed to promote model: {str(e)}")
        raise e

    
if __name__ == "__main__":
    main()