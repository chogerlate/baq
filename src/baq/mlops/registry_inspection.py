from typing import List, Dict, Any
from wandb import Api

def list_models(
        api: Api,
        registry_entity: str = "chogerlate",
        project: str = "wandb-registry-model",
        model_name: str = "baq-forecastors"
    ) -> List[Dict[str, Any]]:
    """
    List models and their metrics from the W&B model registry.
    
    Args:
        api: W&B API object
        registry_entity: W&B entity name
        model_name: Full model name in format "{registry_entity}/{project}/{model_name}"
        
    Returns:
        list: List of model dictionaries with their information and metrics
    """
    models_list = []
    model_name = f"{registry_entity}/{project}/{model_name}"
    try:
        # Get models from registry
        models = api.artifacts(type_name="model", name=model_name, per_page=100)
        for model in models:
            model_info = {
                "name": model.name,
                "version": model.version,
                "type": model.type,
                "description": model.description,
                "created_at": model.created_at,
                "aliases": model.aliases,
                "run_id": None,
                "single_step_metrics": None,
                "multi_step_metrics": None
            }
            
            print(f"Model name: {model.name}")
            print(f"Model version: {model.version}") 
            print(f"Model type: {model.type}")
            print(f"Model description: {model.description}")
            print(f"Model created at: {model.created_at}")
            print(f"Model tags: {model.aliases}")  # Show model tags/aliases
            
            # Get run ID and metrics from the run that created this model
            run = model.logged_by()
            if run:
                model_info["run_id"] = run.id
                print(f"Run ID: {run.id}")
                
                if 'single_step_metrics' in run.summary:
                    metrics = run.summary['single_step_metrics']
                    model_info["single_step_metrics"] = metrics
                    print("Single-step metrics:")
                    print(f"  MAE: {metrics['mae']:.6f}")
                    print(f"  MAPE: {metrics['mape']:.6f}")
                    
                if 'multi_step_metrics' in run.summary:
                    metrics = run.summary['multi_step_metrics'] 
                    model_info["multi_step_metrics"] = metrics
                    print("Multi-step metrics:")
                    print(f"  MAE: {metrics['mae']:.6f}")
                    print(f"  MAPE: {metrics['mape']:.6f}")
            
            models_list.append(model_info)
            print("---")
            
        return models_list
        
    except Exception as e:
        print(f"Error: {e}")
        return []
