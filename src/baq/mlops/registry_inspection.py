from typing import List, Dict, Any
from wandb import Api

def list_models(
        api: Api,
        collection_path: str = "chogerlate/wandb-registry-model/baq-forecastors"
    ) -> List[Dict[str, Any]]:
    """
    List models and their metrics from the W&B model registry.
    
    Args:
        api: W&B API object
        collection_path: Full model name in format "{entity}/{project}/{model_name}"
        
    Returns:
        list: List of model dictionaries with their information and metrics
    """
    models_list = []
    try:
        # Get models from registry
        models = api.artifacts(type_name="model", name=collection_path, per_page=100)
        for model in models:
            model_info = {
                "name": model.name,
                "version": model.version,
                "aliases": model.aliases,
                "created_at": model.created_at,
                "description": model.description,
                "run": None,
                "metrics": None
            }
            # Get run and metrics from the run that created this model
            run = model.logged_by()
            if run:
                run_info = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at
                }
                # Extract metrics
                single_step = run.summary.get("single_step_metrics", {})
                multi_step = run.summary.get("multi_step_metrics", {})
                metrics = {
                    "single_step_accuracy": single_step.get("accuracy"),
                    "single_step_mape": single_step.get("mape"),
                    "single_step_mae": single_step.get("mae"),
                    "multi_step_accuracy": multi_step.get("accuracy"),
                    "multi_step_mape": multi_step.get("mape"),
                    "multi_step_mae": multi_step.get("mae"),
                    "runtime": run.summary.get("_runtime")
                }
                model_info["run"] = run_info
                model_info["metrics"] = metrics
            models_list.append(model_info)
        return models_list
    except Exception as e:
        print(f"Error: {e}")
        return []
