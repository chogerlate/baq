import os
import wandb
import argparse
from datetime import datetime
from typing import List, Dict, Any

def get_registration_emoji(model_artifacts_count):
    """Get emoji based on model registration status"""
    if model_artifacts_count > 0:
        return "✅"  # Model registered
    else:
        return "❌"  # Model not registered

def get_model_registration_status(model_artifacts_count, model_names=None):
    """Get detailed model registration status"""
    if model_artifacts_count == 0:
        return "❌ Not registered"
    elif model_artifacts_count == 1:
        model_name = model_names[0] if model_names else "model"
        return f"✅ Registered (1 model: {model_name})"
    else:
        return f"✅ Registered ({model_artifacts_count} models)"

def get_registration_status(artifacts_count, model_artifacts):
    """Get detailed registration status"""
    if artifacts_count == 0:
        return "❌ Not Registered"
    elif model_artifacts > 0:
        return f"📦 Registered ({model_artifacts} models, {artifacts_count} total)"
    else:
        return f"📦 Registered ({artifacts_count} artifacts, no models)"

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

def get_status_emoji(state):
    """Get emoji based on run state"""
    status_map = {
        "finished": "✅",
        "running": "🏃",
        "failed": "❌",
        "crashed": "💥",
        "killed": "⚰️",
        "preempted": "⏸️"
    }
    return status_map.get(state, "❓")

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

def format_date(date_value):
    """Format date in a readable way"""
    if not date_value:
        return "Unknown"
    if isinstance(date_value, str):
        try:
            from datetime import datetime
            date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except:
            return date_value
    try:
        return date_value.strftime('%Y-%m-%d %H:%M')
    except:
        return str(date_value)

def get_metrics_from_run(run):
    """Extract metrics from run summary following the user's pattern"""
    if not run or not run.summary:
        return {
            "single_step_accuracy": None,
            "single_step_mape": None,
            "single_step_mae": None,
            "multi_step_accuracy": None,
            "multi_step_mape": None,
            "multi_step_mae": None,
            "runtime": None
        }
    
    # Only process runs that have both single_step_metrics and multi_step_metrics
    if 'single_step_metrics' not in run.summary or 'multi_step_metrics' not in run.summary:
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

def list_runs(api: wandb.Api, entity: str, project: str, limit: int = 50, filters: Dict = None) -> List[Dict[str, Any]]:
    """
    List runs from a W&B project with their metrics and information.
    Following the user's pattern for querying runs.
    
    Args:
        api: W&B API object
        entity: W&B entity/username
        project: W&B project name
        limit: Maximum number of runs to fetch
        filters: Optional filters for runs
        
    Returns:
        list: List of run dictionaries with their information and metrics
    """
    runs_list = []
    try:
        # Get runs from project using the user's pattern
        project_path = f"{entity}/{project}"
        print(f"Fetching runs from project: {project_path}")
        
        # Get runs from this project
        runs = api.runs(project_path, filters=filters, per_page=limit)
        
        print("Processing runs...")
        for run in runs:
            # Only process runs that have summary data with metrics (following user's pattern)
            if run.summary and 'single_step_metrics' in run.summary and 'multi_step_metrics' in run.summary:
                print(f"Processing run {run.name} ({run.id})")
                
                # Extract basic run information
                run_info = {
                    "id": run.id,
                    "name": run.name,
                    "display_name": run.display_name or run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "user": run.user.username if run.user else "Unknown",
                    "tags": run.tags,
                    "config": dict(run.config) if run.config else {},
                    "url": run.url
                }
                
                # Extract metrics using the updated function
                metrics = get_metrics_from_run(run)
                run_info["metrics"] = metrics
                
                # Get artifacts count and model registration info
                try:
                    artifacts = list(run.logged_artifacts())
                    model_artifacts = [artifact for artifact in artifacts if artifact.type == "model"]
                    
                    run_info["artifacts_count"] = len(artifacts)
                    run_info["model_artifacts"] = len(model_artifacts)
                    
                    # Collect model registration details
                    if model_artifacts:
                        model_details = []
                        for model in model_artifacts:
                            model_details.append(f"{model.name}:{model.version}")
                        run_info["model_details"] = model_details
                        run_info["model_registration"] = f"✅ Registered ({len(model_artifacts)} models)"
                        print(f"  Model registration: ✅ Registered ({len(model_artifacts)} models)")
                        for model in model_artifacts:
                            print(f"    - {model.name}:{model.version}")
                    else:
                        run_info["model_details"] = []
                        run_info["model_registration"] = "❌ Not registered"
                        print(f"  Model registration: ❌ Not registered")
                        
                except Exception as e:
                    run_info["artifacts_count"] = 0
                    run_info["model_artifacts"] = 0
                    run_info["model_details"] = []
                    run_info["model_registration"] = f"❓ Error checking: {e}"
                    print(f"  Model registration: ❓ Error checking: {e}")
                
                runs_list.append(run_info)
            else:
                print(f"Skipping run {run.name} ({run.id}) - no metrics data")
            
        print(f"Found {len(runs_list)} runs with metrics data")
        return runs_list
        
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

def generate_markdown_report(runs_data: List[Dict], entity: str, project: str, filters: Dict = None):
    """Generate a concise markdown report from runs data with only essential information."""
    
    with open("runs_report.md", "w") as f:
        f.write(f"# 🏃 W&B Runs Report: {entity}/{project}\n\n")
        f.write(f"**Total Runs:** {len(runs_data)} | **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        if not runs_data:
            f.write("❌ No runs found in project.\n\n")
            return

        # Quick Summary Stats
        registered_runs = [run for run in runs_data if run['model_artifacts'] > 0]
        unregistered_runs = [run for run in runs_data if run['model_artifacts'] == 0]
        finished_runs = [run for run in runs_data if run['state'] == 'finished']
        
        f.write("## 📊 Quick Summary\n\n")
        f.write(f"- ✅ **Finished**: {len(finished_runs)}/{len(runs_data)} runs\n")
        f.write(f"- 📦 **Registered**: {len(registered_runs)}/{len(runs_data)} runs ({len(registered_runs)/len(runs_data)*100:.0f}%)\n")
        f.write(f"- 🤖 **Total Models**: {sum(run['model_artifacts'] for run in runs_data)}\n\n")

        # Top 10 Performance Leaderboard (most important section)
        f.write("## 🏆 Top 10 Performance Leaderboard\n\n")
        f.write("| Rank | Run Name | Run ID | Single-Acc | Multi-Acc | Models | Status |\n")
        f.write("|------|----------|--------|------------|-----------|--------|--------|\n")

        # Sort runs by performance
        sorted_runs = sorted(
            runs_data,
            key=lambda x: (
                x['metrics']['single_step_accuracy'] or 0,
                x['created_at'] or datetime.min
            ),
            reverse=True
        )

        for i, run in enumerate(sorted_runs[:10], 1):
            metrics = run['metrics'] or {}
            status_emoji = get_status_emoji(run['state'])
            reg_emoji = get_registration_emoji(run['model_artifacts'])
            
            # Format metrics
            single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
            multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
            
            # Compose run name with link
            run_name = f"[{run['display_name']}]({run['url']})"
            
            f.write(f"| {i} | {run_name} | `{run['id']}` | {single_acc} | {multi_acc} | {reg_emoji}{run['model_artifacts']} | {status_emoji}{run['state']} |\n")

        # Problem Runs (Failed/Crashed) - Important for debugging
        problem_runs = [run for run in runs_data if run['state'] in ['failed', 'crashed', 'killed']]
        if problem_runs:
            f.write(f"\n## ⚠️ Problem Runs ({len(problem_runs)})\n\n")
            f.write("| Run Name | Run ID | Status | User | Created |\n")
            f.write("|----------|--------|--------|------|--------|\n")
            
            for run in problem_runs[:5]:  # Show only first 5
                status_emoji = get_status_emoji(run['state'])
                created = format_date(run['created_at'])
                run_link = f"[{run['display_name']}]({run['url']})"
                
                f.write(f"| {run_link} | `{run['id']}` | {status_emoji}{run['state']} | {run['user']} | {created} |\n")
            
            if len(problem_runs) > 5:
                f.write(f"\n*... and {len(problem_runs) - 5} more problem runs*\n")

        # Unregistered High-Performance Runs (Action needed)
        high_perf_unregistered = [
            run for run in unregistered_runs 
            if run['metrics']['single_step_accuracy'] is not None and run['metrics']['single_step_accuracy'] > 0.8
        ]
        
        if high_perf_unregistered:
            f.write(f"\n## 🎯 High-Performance Unregistered Runs ({len(high_perf_unregistered)})\n")
            f.write("*These runs have good performance but no registered models - consider registering them*\n\n")
            f.write("| Run Name | Run ID | Single-Acc | Multi-Acc | User |\n")
            f.write("|----------|--------|------------|-----------|------|\n")
            
            for run in sorted(high_perf_unregistered, key=lambda x: x['metrics']['single_step_accuracy'], reverse=True)[:5]:
                metrics = run['metrics']
                single_acc = f"{metrics['single_step_accuracy']:.3f}"
                multi_acc = f"{metrics['multi_step_accuracy']:.3f}" if metrics['multi_step_accuracy'] is not None else "N/A"
                run_link = f"[{run['display_name']}]({run['url']})"
                
                f.write(f"| {run_link} | `{run['id']}` | {single_acc} | {multi_acc} | {run['user']} |\n")

        # Usage Instructions (simplified)
        f.write("\n## 🔧 Quick Actions\n\n")
        f.write("- **Register a model**: Use run ID from high-performance unregistered runs\n")
        f.write("- **Debug failures**: Check problem runs section for failed experiments\n")
        f.write("- **Compare models**: Use performance leaderboard for model selection\n")

def main():
    parser = argparse.ArgumentParser(description="Generate W&B Runs Report")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "chogerlate"), 
                        help="W&B entity/username")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "test-pipeline"), 
                        help="W&B project name")
    parser.add_argument("--limit", type=int, default=50, 
                        help="Maximum number of runs to fetch")
    parser.add_argument("--state", 
                        help="Filter by run state (finished, running, failed, etc.)")
    parser.add_argument("--tags", 
                        help="Filter by tags (comma-separated)")
    
    args = parser.parse_args()
    
    # Build filters
    filters = {}
    if args.state:
        filters["state"] = args.state
    if args.tags:
        filters["tags"] = {"$in": args.tags.split(",")}
    
    api = wandb.Api()
    runs_data = list_runs(
        api,
        entity=args.entity,
        project=args.project,
        limit=args.limit,
        filters=filters if filters else None
    )
    
    generate_markdown_report(runs_data, args.entity, args.project, filters if filters else None)
    print(f"✅ Runs report generated successfully: runs_report.md")

if __name__ == "__main__":
    main()
