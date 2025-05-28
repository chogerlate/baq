import os
import wandb
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any

def get_performance_emoji(accuracy):
    """Get emoji based on accuracy performance"""
    if accuracy is None:
        return "❓"
    if accuracy >= 0.95:
        return "🏆"
    elif accuracy >= 0.90:
        return "🥇"
    elif accuracy >= 0.85:
        return "🥈"
    elif accuracy >= 0.80:
        return "🥉"
    elif accuracy >= 0.70:
        return "👍"
    else:
        return "⚠️"

def format_runtime(runtime_seconds):
    """Format runtime in a readable way"""
    if runtime_seconds is None:
        return "N/A"
    if runtime_seconds < 60:
        return f"{runtime_seconds:.1f}s"
    elif runtime_seconds < 3600:
        return f"{runtime_seconds/60:.1f}m"
    else:
        return f"{runtime_seconds/3600:.1f}h"

def get_metrics_from_run(run):
    """Extract metrics from run summary"""
    if not run:
        return {
            "single_step_accuracy": None,
            "single_step_mape": None,
            "single_step_mae": None,
            "multi_step_accuracy": None,
            "multi_step_mape": None,
            "multi_step_mae": None,
            "runtime": None
        }
    
    single_step = run.summary.get("single_step_metrics", {})
    multi_step = run.summary.get("multi_step_metrics", {})
    
    return {
        "single_step_accuracy": single_step.get("accuracy"),
        "single_step_mape": single_step.get("mape"),
        "single_step_mae": single_step.get("mae"),
        "multi_step_accuracy": multi_step.get("accuracy"),
        "multi_step_mape": multi_step.get("mape"),
        "multi_step_mae": multi_step.get("mae"),
        "runtime": run.summary.get("_runtime")
    }

def get_alias_status(aliases):
    """Get deployment status from aliases"""
    status = []
    if not aliases:
        return "📦 unregistered"
    
    if any("production" in alias.lower() for alias in aliases):
        status.append("🚀 production")
    if any("staging" in alias.lower() for alias in aliases):
        status.append("🔍 staging")
    if any(("latest" in alias.lower() and "staging" not in alias.lower() and "production" not in alias.lower()) for alias in aliases):
        status.append("📦 latest")
        
    return " | ".join(status) if status else "📦 registered"

def get_model_registry_report(
        api,
        collection_path: str
    ):
    """
    Generate a comprehensive report of models in the W&B Model Registry.
    
    Args:
        api: W&B API object
        registry_entity: W&B entity name for the registry
        project: W&B project name
        model_name: Model name in the registry
        
    Returns:
        List[Dict]: List of model information dictionaries
    """
    models_data = []
    
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
                "status": get_alias_status(model.aliases),
                "run": None,
                "metrics": None
            }
            
            # Get run that created this model
            run = model.logged_by()
            if run:
                model_info["run"] = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at
                }
                model_info["metrics"] = get_metrics_from_run(run)
            
            models_data.append(model_info)
        
        return models_data
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def generate_markdown_report(models_data, collection_path):
    """Generate a markdown report from model registry data"""
    
    with open("model_registry_report.md", "w") as f:
        # Header
        f.write(f"# 🏆 Model Registry Report: {collection_path}\n\n")
        f.write(f"**Registry Path:** `{collection_path}`\n")
        f.write(f"**Total Models:** {len(models_data)}\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        
        if not models_data:
            f.write("❌ No models found in registry.\n\n")
            return
        
        # Find models with production/staging aliases
        production_models = [m for m in models_data if "production" in " ".join(m["aliases"]).lower()]
        staging_models = [m for m in models_data if "staging" in " ".join(m["aliases"]).lower()]
        
        # Summary section
        f.write("## 📊 Registry Summary\n\n")
        
        if production_models:
            prod_model = production_models[0]  # Most recent production model
            f.write(f"### 🚀 Current Production Model\n\n")
            f.write(f"**Version:** v{prod_model['version']}\n")
            if prod_model['run']:
                f.write(f"**Model Name:** {prod_model['run']['name']}\n")
                f.write(f"**Run ID:** `{prod_model['run']['id']}`\n")
            f.write(f"**Deployed:** {prod_model['created_at'].strftime('%Y-%m-%d')}\n")
            
            # Show metrics if available
            if prod_model['metrics'] and prod_model['metrics']['single_step_accuracy']:
                f.write("\n**Performance Metrics:**\n")
                f.write(f"- Single-step Accuracy: {prod_model['metrics']['single_step_accuracy']:.4f}\n")
                f.write(f"- Multi-step Accuracy: {prod_model['metrics']['multi_step_accuracy']:.4f}\n")
                f.write(f"- Single-step MAPE: {prod_model['metrics']['single_step_mape']:.4f}%\n")
                f.write(f"- Multi-step MAPE: {prod_model['metrics']['multi_step_mape']:.4f}%\n")
            f.write("\n")
        
        if staging_models:
            staging_model = staging_models[0]  # Most recent staging model
            f.write(f"### 🔍 Current Staging Model\n\n")
            f.write(f"**Version:** v{staging_model['version']}\n")
            if staging_model['run']:
                f.write(f"**Model Name:** {staging_model['run']['name']}\n")
                f.write(f"**Run ID:** `{staging_model['run']['id']}`\n")
            f.write(f"**Staged:** {staging_model['created_at'].strftime('%Y-%m-%d')}\n")
            
            # Show metrics if available
            if staging_model['metrics'] and staging_model['metrics']['single_step_accuracy']:
                f.write("\n**Performance Metrics:**\n")
                f.write(f"- Single-step Accuracy: {staging_model['metrics']['single_step_accuracy']:.4f}\n")
                f.write(f"- Multi-step Accuracy: {staging_model['metrics']['multi_step_accuracy']:.4f}\n")
                f.write(f"- Single-step MAPE: {staging_model['metrics']['single_step_mape']:.4f}%\n")
                f.write(f"- Multi-step MAPE: {staging_model['metrics']['multi_step_mape']:.4f}%\n")
            f.write("\n")
        
        # Model leaderboard
        f.write("## 📈 Model Performance Leaderboard\n\n")
        f.write("| Version | Status | Model | Single-Acc | Multi-Acc | Single-MAPE | Multi-MAPE | Created |\n")
        f.write("|---------|--------|-------|------------|-----------|-------------|------------|--------|\n")
        
        # Sort models by performance (if metrics available) or version
        sorted_models = sorted(
            models_data, 
            key=lambda x: (x['metrics']['single_step_accuracy'] or 0) if x['metrics'] else 0, 
            reverse=True
        )
        
        for model in sorted_models:
            metrics = model['metrics'] or {}
            model_name = model['run']['name'] if model['run'] else "Unknown"
            created_date = model['created_at'].strftime('%Y-%m-%d') if model['created_at'] else "Unknown"
            
            # Format metrics
            single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
            multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
            single_mape = f"{metrics.get('single_step_mape', 0):.2f}%" if metrics.get('single_step_mape') is not None else "N/A"
            multi_mape = f"{metrics.get('multi_step_mape', 0):.2f}%" if metrics.get('multi_step_mape') is not None else "N/A"
            
            # Add performance emoji
            perf_emoji = get_performance_emoji(metrics.get('single_step_accuracy'))
            
            f.write(f"| v{model['version']} | {model['status']} | {perf_emoji} {model_name} | {single_acc} | {multi_acc} | {single_mape} | {multi_mape} | {created_date} |\n")
        
        # Detailed model analysis
        f.write("\n## 🔍 Detailed Model Analysis\n\n")
        
        for model in sorted_models:
            metrics = model['metrics'] or {}
            status = model['status']
            model_name = model['run']['name'] if model['run'] else "Unknown"
            run_id = model['run']['id'] if model['run'] else "Unknown"
            
            f.write(f"### Model v{model['version']} ({status})\n\n")
            
            # Model information
            f.write("**📋 Model Information:**\n")
            f.write(f"- **Name:** {model_name}\n")
            f.write(f"- **Version:** v{model['version']}\n")
            f.write(f"- **Aliases:** {', '.join(model['aliases']) if model['aliases'] else 'None'}\n")
            f.write(f"- **Created:** {model['created_at']}\n")
            if model['run']:
                f.write(f"- **Run ID:** `{run_id}`\n")
                f.write(f"- **Run Status:** {model['run']['state']}\n")
            f.write("\n")
            
            # Metrics if available
            if metrics and metrics.get('single_step_accuracy') is not None:
                f.write("**📊 Performance Metrics:**\n\n")
                
                f.write("*Single-Step Prediction:*\n")
                f.write(f"- **Accuracy:** {metrics.get('single_step_accuracy', 'N/A'):.4f}\n")
                f.write(f"- **MAPE:** {metrics.get('single_step_mape', 'N/A'):.4f}%\n")
                f.write(f"- **MAE:** {metrics.get('single_step_mae', 'N/A'):.4f}\n\n")
                
                f.write("*Multi-Step Prediction:*\n")
                f.write(f"- **Accuracy:** {metrics.get('multi_step_accuracy', 'N/A'):.4f}\n")
                f.write(f"- **MAPE:** {metrics.get('multi_step_mape', 'N/A'):.4f}%\n")
                f.write(f"- **MAE:** {metrics.get('multi_step_mae', 'N/A'):.4f}\n\n")
                
                # Training time
                if metrics.get('runtime'):
                    f.write(f"**⏱️ Training Time:** {format_runtime(metrics['runtime'])}\n\n")
            else:
                f.write("**⚠️ No metrics available for this model**\n\n")
            
            # Quick actions
            f.write("**🔗 Quick Actions:**\n")
            artifact_url = f"https://wandb.ai/{collection_path}/artifacts/{model['name']}/{model['version']}"
            f.write(f"- [📦 View Model in Registry]({artifact_url})\n")
            if model['run']:
                run_url = f"https://wandb.ai/{collection_path}/runs/{run_id}"
                f.write(f"- [📊 View Original Run]({run_url})\n")
            
            # Promotion/deployment options
            f.write("\n**🚀 Deployment Options:**\n")
            f.write(f"- `/promote {run_id} staging` - Promote to staging\n")
            f.write(f"- `/promote {run_id} production` - Promote to production\n")
            
            f.write("\n---\n\n")
        
        # Usage instructions
        f.write("## 📚 How to Use These Models\n\n")
        f.write("```python\n")
        f.write("import wandb\n")
        f.write("import joblib\n\n")
        f.write("# Load production model\n")
        f.write(f"api = wandb.Api()\n")
        f.write(f"artifact = api.artifact('{collection_path}:production', type='model')\n")
        f.write("model_dir = artifact.download()\n")
        f.write("model = joblib.load(f'{model_dir}/model')\n\n")
        f.write("# Make predictions\n")
        f.write("predictions = model.predict(X_test)\n")
        f.write("```\n\n")
        
        # Footer
        f.write("---\n")
        f.write(f"*Report generated by W&B Model Registry Inspector at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC* 🤖\n")

def main():
    parser = argparse.ArgumentParser(description="Generate W&B Model Registry Report")
    parser.add_argument("--collection-path", default=os.environ.get("WANDB_MODEL_REGISTRY_COLLECTION_PATH", "chogerlate/wandb-registry-model/baq-forecastors"), 
                        help="W&B registry collection path")
    args = parser.parse_args()
    
    api = wandb.Api()
    models_data = get_model_registry_report(
        api,
        collection_path=args.collection_path
    )
    
    generate_markdown_report(models_data, args.collection_path)
    print(f"✅ Model registry report generated successfully: model_registry_report.md")

if __name__ == "__main__":
    main()
