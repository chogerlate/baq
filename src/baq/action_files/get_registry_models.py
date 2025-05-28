import os
import wandb
import argparse
from datetime import datetime
from typing import List, Dict, Any

from baq.mlops.registry_inspection import list_models

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

def get_alias_status(aliases, version):
    """Get deployment status from aliases"""
    status = []
    if not aliases:
        return f"📦 {version}"
    
    # Define important aliases to look for
    important_aliases = ["production", "staging", "latest"]
    
    # Check for production alias
    production_aliases = [alias for alias in aliases if "production" in alias.lower()]
    if production_aliases:
        status.append("🚀 production")
    
    # Check for staging alias
    staging_aliases = [alias for alias in aliases if "staging" in alias.lower()]
    if staging_aliases:
        status.append("🔍 staging")
    
    # Check for latest alias (but not staging-latest or production-latest)
    latest_aliases = [alias for alias in aliases if "latest" in alias.lower() 
                     and "staging" not in alias.lower() 
                     and "production" not in alias.lower()]
    if latest_aliases:
        status.append("📦 latest")
    
    return ", ".join(status)

def generate_markdown_report(models_data, collection_path):
    """Generate a concise markdown report from model registry data with only essential information."""
    
    # Helper function to safely format dates
    def format_date(date_value):
        if not date_value:
            return "Unknown"
        if isinstance(date_value, str):
            try:
                from datetime import datetime
                date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except:
                return date_value
        try:
            return date_value.strftime('%Y-%m-%d')
        except:
            return str(date_value)

    with open("model_registry_report.md", "w") as f:
        f.write(f"# 🏆 Model Registry: {collection_path}\n\n")
        f.write(f"**Total Models:** {len(models_data)} | **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        if not models_data:
            f.write("❌ No models found in registry.\n\n")
            return

        # Quick Summary
        production_models = [m for m in models_data if 'production' in [alias.lower() for alias in m['aliases']]]
        staging_models = [m for m in models_data if 'staging' in [alias.lower() for alias in m['aliases']]]
        
        f.write("## 📊 Quick Summary\n\n")
        f.write(f"- 🚀 **Production Models**: {len(production_models)}\n")
        f.write(f"- 🧪 **Staging Models**: {len(staging_models)}\n")
        f.write(f"- 📦 **Total Versions**: {len(models_data)}\n\n")

        # Top 10 Performance Leaderboard
        f.write("## 🏆 Top 10 Model Performance\n\n")
        f.write("| Rank | Model Name | Version | Single-Acc | Multi-Acc | Status | Created |\n")
        f.write("|------|------------|---------|------------|-----------|--------|--------|\n")

        # Sort models by performance
        sorted_models = sorted(
            models_data,
            key=lambda x: (x['metrics']['single_step_accuracy'] or 0) if x['metrics'] else 0,
            reverse=True
        )

        for i, model in enumerate(sorted_models[:10], 1):
            metrics = model['metrics'] or {}
            model_name = model['run']['name'] if model['run'] else "Unknown"
            version = model['version']
            created_date = format_date(model['created_at'])
            status = get_alias_status(model['aliases'], version)
            
            single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
            multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
            
            f.write(f"| {i} | {model_name} | v{version} | {single_acc} | {multi_acc} | {status} | {created_date} |\n")

        # Production Models (most important)
        if production_models:
            f.write(f"\n## 🚀 Production Models ({len(production_models)})\n\n")
            f.write("| Model Name | Version | Single-Acc | Multi-Acc | Created |\n")
            f.write("|------------|---------|------------|-----------|--------|\n")
            
            for model in sorted(production_models, key=lambda x: (x['metrics']['single_step_accuracy'] or 0) if x['metrics'] else 0, reverse=True):
                metrics = model['metrics'] or {}
                model_name = model['run']['name'] if model['run'] else "Unknown"
                version = model['version']
                created_date = format_date(model['created_at'])
                
                single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
                multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
                
                f.write(f"| {model_name} | v{version} | {single_acc} | {multi_acc} | {created_date} |\n")

        # Staging Models Ready for Promotion
        if staging_models:
            f.write(f"\n## 🧪 Staging Models ({len(staging_models)})\n")
            f.write("*Models ready for production consideration*\n\n")
            f.write("| Model Name | Version | Single-Acc | Multi-Acc | Created |\n")
            f.write("|------------|---------|------------|-----------|--------|\n")
            
            for model in sorted(staging_models, key=lambda x: (x['metrics']['single_step_accuracy'] or 0) if x['metrics'] else 0, reverse=True):
                metrics = model['metrics'] or {}
                model_name = model['run']['name'] if model['run'] else "Unknown"
                version = model['version']
                created_date = format_date(model['created_at'])
                
                single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
                multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
                
                f.write(f"| {model_name} | v{version} | {single_acc} | {multi_acc} | {created_date} |\n")

        # Quick Actions
        f.write("\n## 🔧 Quick Actions\n\n")
        f.write("- **Promote to Production**: Select high-performing staging models\n")
        f.write("- **Compare Performance**: Use leaderboard for model selection\n")
        f.write("- **Monitor Production**: Check production model performance\n")

def main():
    parser = argparse.ArgumentParser(description="Generate W&B Model Registry Report")
    parser.add_argument("--collection-path", default=os.environ.get("WANDB_MODEL_REGISTRY_COLLECTION_PATH", "chogerlate/wandb-registry-model/baq-forecastors"), 
                        help="W&B registry collection path")
    args = parser.parse_args()
    
    api = wandb.Api()
    models_data = list_models(
        api,
        collection_path=args.collection_path
    )
    
    generate_markdown_report(models_data, args.collection_path)
    print(f"✅ Model registry report generated successfully: model_registry_report.md")

if __name__ == "__main__":
    main()
