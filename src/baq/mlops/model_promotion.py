
from typing import Optional, Dict, Any, Union
from wandb import Api, util
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


def promote_model(
    model_name: str,
    version: str,
    add_aliases: Optional[Union[str, list]] = None,
    remove_aliases: Optional[Union[str, list]] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    description: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Promote a model version by adding/removing multiple aliases in Weights & Biases.
    
    Args:
        model_name (str): Name of the model in the model registry
        version (str): Version identifier (e.g., "v1", "v2", or version number)
        add_aliases (str or list, optional): Alias(es) to add (e.g., "staging" or ["staging", "production"])
        remove_aliases (str or list, optional): Alias(es) to remove (e.g., "old" or ["old", "deprecated"])
        entity (str, optional): WandB entity (username/team). If None, uses default
        project (str, optional): WandB project name. If None, uses default
        description (str, optional): Description for the promotion
    
    Returns:
        Dict[str, Any]: Result dictionary with promotion details
        
    Raises:
        ValueError: If required parameters are missing or invalid
        wandb.Error: If WandB API operations fail
    """
    
    # Validate inputs
    if not model_name or not version:
        raise ValueError("model_name and version are required")
    
    if not add_aliases and not remove_aliases:
        raise ValueError("At least one of add_aliases or remove_aliases must be provided")
    
    # Convert string inputs to lists
    if isinstance(add_aliases, str):
        add_aliases = [add_aliases]
    elif add_aliases is None:
        add_aliases = []
    
    if isinstance(remove_aliases, str):
        remove_aliases = [remove_aliases]
    elif remove_aliases is None:
        remove_aliases = []
    
    try:
        # Initialize wandb API
        api = Api(api_key=api_key)
        
        # Construct model path
        if entity and project:
            model_path = f"{entity}/{project}/{model_name}:{version}"
        elif entity:
            model_path = f"{entity}/{model_name}:{version}"
        else:
            model_path = f"{model_name}:{version}"
        
        # Get the model version
        model_version = api.artifact(model_path)
        
        # Get current aliases
        current_aliases = model_version.aliases.copy()
        
        logging.info(f"Current aliases for {model_path}: {current_aliases}")
        
        # Handle alias updates
        new_aliases = current_aliases.copy()
        
        # Remove aliases
        removed_aliases = []
        for alias in remove_aliases:
            if alias in new_aliases:
                new_aliases.remove(alias)
                removed_aliases.append(alias)
                logging.info(f"Will remove alias '{alias}' from {model_path}")
            else:
                logging.warning(f"Alias '{alias}' not found in current aliases, skipping removal")
        
        # Add new aliases
        added_aliases = []
        for alias in add_aliases:
            if alias not in new_aliases:
                new_aliases.append(alias)
                added_aliases.append(alias)
                logging.info(f"Will add alias '{alias}' to {model_path}")
            else:
                logging.warning(f"Alias '{alias}' already exists, skipping addition")
        
        # Update aliases by completely replacing them
        model_version.aliases = new_aliases
        
        # Update description if provided
        if description:
            model_version.description = description
        
        # Save all changes
        model_version.save()
        
        logging.info(f"Successfully updated aliases for {model_path}")
        
        # Prepare result
        result = {
            "success": True,
            "model_path": model_path,
            "version": version,
            "added_aliases": added_aliases,
            "removed_aliases": removed_aliases,
            "previous_aliases": current_aliases,
            "current_aliases": model_version.aliases,
            "description": description,
            "promoted_at": util.generate_id()
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to promote model {model_name}:{version} - {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "version": version,
            "add_aliases": add_aliases,
            "remove_aliases": remove_aliases
        }
