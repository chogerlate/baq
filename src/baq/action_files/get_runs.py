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
    """Generate a markdown report from runs data."""
    
    with open("runs_report.md", "w") as f:
        f.write(f"# 🏃 W&B Runs Report: {entity}/{project}\n\n")
        f.write(f"**Project Path:** `{entity}/{project}`\n")
        f.write(f"**Total Runs:** {len(runs_data)}\n")
        if filters:
            f.write(f"**Filters Applied:** {filters}\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        
        if not runs_data:
            f.write("❌ No runs found in project.\n\n")
            return

        # Performance Leaderboard
        f.write("## 📈 Run Performance Leaderboard\n\n")
        f.write("| Status | Registration | Run Name | Single-Acc | Multi-Acc | Single-MAPE | Multi-MAPE | Runtime | Created |\n")
        f.write("|--------|--------------|----------|------------|-----------|-------------|------------|---------|--------|\n")

        # Sort runs by performance (single step accuracy) or creation date
        sorted_runs = sorted(
            runs_data,
            key=lambda x: (
                x['metrics']['single_step_accuracy'] or 0,
                x['created_at'] or datetime.min
            ),
            reverse=True
        )

        for run in sorted_runs:
            metrics = run['metrics'] or {}
            status_emoji = get_status_emoji(run['state'])
            perf_emoji = get_performance_emoji(metrics.get('single_step_accuracy'))
            reg_emoji = get_registration_emoji(run['model_artifacts'])
            
            # Format metrics
            single_acc = f"{metrics.get('single_step_accuracy', 0):.3f}" if metrics.get('single_step_accuracy') is not None else "N/A"
            multi_acc = f"{metrics.get('multi_step_accuracy', 0):.3f}" if metrics.get('multi_step_accuracy') is not None else "N/A"
            single_mape = f"{metrics.get('single_step_mape', 0):.2f}%" if metrics.get('single_step_mape') is not None else "N/A"
            multi_mape = f"{metrics.get('multi_step_mape', 0):.2f}%" if metrics.get('multi_step_mape') is not None else "N/A"
            runtime = format_runtime(metrics.get('runtime'))
            created = format_date(run['created_at'])
            
            # Compose run name with emojis
            full_run_name = f"{perf_emoji} [{run['display_name']}]({run['url']})"
            
            f.write(f"| {status_emoji} {run['state']} | {reg_emoji} {run['model_artifacts']} | {full_run_name} | {single_acc} | {multi_acc} | {single_mape} | {multi_mape} | {runtime} | {created} |\n")

        # Registration Status Summary
        f.write("\n## 📦 Registration Status Summary\n\n")
        registered_runs = [run for run in runs_data if run['model_artifacts'] > 0]
        unregistered_runs = [run for run in runs_data if run['model_artifacts'] == 0]
        
        f.write(f"- 📦 **Registered Runs**: {len(registered_runs)} / {len(runs_data)} ({len(registered_runs)/len(runs_data)*100:.1f}%)\n")
        f.write(f"- ❌ **Unregistered Runs**: {len(unregistered_runs)} / {len(runs_data)} ({len(unregistered_runs)/len(runs_data)*100:.1f}%)\n")
        
        total_models = sum(run['model_artifacts'] for run in runs_data)
        f.write(f"- 🤖 **Total Models**: {total_models}\n\n")
        
        # Unregistered Runs Details
        if unregistered_runs:
            f.write("### ❌ Unregistered Runs (No Models)\n\n")
            f.write("| Status | Run Name | User | Created | State |\n")
            f.write("|--------|----------|------|---------|-------|\n")
            
            for run in unregistered_runs[:10]:  # Show first 10
                status_emoji = get_status_emoji(run['state'])
                created = format_date(run['created_at'])
                run_link = f"[{run['display_name']}]({run['url']})"
                
                f.write(f"| {status_emoji} | {run_link} | {run['user']} | {created} | {run['state']} |\n")
            
            if len(unregistered_runs) > 10:
                f.write(f"\n*... and {len(unregistered_runs) - 10} more unregistered runs*\n")
            f.write("\n")
        
        # Registered Runs Details
        if registered_runs:
            f.write("### 🤖 Registered Runs (With Models)\n\n")
            f.write("| Status | Run Name | Model Registration Status | User | Created |\n")
            f.write("|--------|----------|---------------------------|------|--------|\n")
            
            # Sort by model count (most models first)
            registered_runs_sorted = sorted(registered_runs, key=lambda x: x['model_artifacts'], reverse=True)
            
            for run in registered_runs_sorted[:10]:  # Show first 10
                status_emoji = get_status_emoji(run['state'])
                created = format_date(run['created_at'])
                run_link = f"[{run['display_name']}]({run['url']})"
                
                # Use the detailed model registration status
                reg_status = run['model_registration']
                
                f.write(f"| {status_emoji} | {run_link} | {reg_status} | {run['user']} | {created} |\n")
            
            if len(registered_runs) > 10:
                f.write(f"\n*... and {len(registered_runs) - 10} more registered runs*\n")
            f.write("\n")
            
            # Show detailed model information for top registered runs
            f.write("#### 📋 Model Details for Top Registered Runs\n\n")
            for i, run in enumerate(registered_runs_sorted[:5], 1):
                f.write(f"**{i}. {run['display_name']}** ({run['id']})\n")
                if run['model_details']:
                    for model_detail in run['model_details']:
                        f.write(f"  - `{model_detail}`\n")
                f.write("\n")

        # Run Status Summary
        f.write("\n## 📊 Run Status Summary\n\n")
        status_counts = {}
        for run in runs_data:
            state = run['state']
            status_counts[state] = status_counts.get(state, 0) + 1
        
        for state, count in sorted(status_counts.items()):
            emoji = get_status_emoji(state)
            f.write(f"- {emoji} **{state.title()}**: {count} runs\n")

        # Top Performers
        f.write("\n## 🏆 Top Performing Runs\n\n")
        top_runs = [run for run in sorted_runs[:5] if run['metrics']['single_step_accuracy'] is not None]
        
        if top_runs:
            for i, run in enumerate(top_runs, 1):
                metrics = run['metrics']
                f.write(f"### {i}. {run['display_name']}\n")
                f.write(f"- **Run ID**: `{run['id']}`\n")
                f.write(f"- **Status**: {get_status_emoji(run['state'])} {run['state']}\n")
                f.write(f"- **User**: {run['user']}\n")
                f.write(f"- **Single-Step Accuracy**: {metrics['single_step_accuracy']:.3f}\n")
                multi_step_acc = f"{metrics['multi_step_accuracy']:.3f}" if metrics['multi_step_accuracy'] is not None else "N/A"
                f.write(f"- **Multi-Step Accuracy**: {multi_step_acc}\n")
                f.write(f"- **Runtime**: {format_runtime(metrics['runtime'])}\n")
                f.write(f"- **Models**: {run['model_artifacts']} models\n")
                if run['tags']:
                    f.write(f"- **Tags**: {', '.join(run['tags'])}\n")
                f.write(f"- **URL**: {run['url']}\n\n")
        else:
            f.write("No runs with performance metrics found.\n\n")

        # Recent Activity
        f.write("## 🕒 Recent Activity\n\n")
        recent_runs = sorted(runs_data, key=lambda x: x['created_at'] or datetime.min, reverse=True)[:10]
        
        f.write("| Status | Run Name | User | Created | State |\n")
        f.write("|--------|----------|------|---------|-------|\n")
        
        for run in recent_runs:
            status_emoji = get_status_emoji(run['state'])
            created = format_date(run['created_at'])
            run_link = f"[{run['display_name']}]({run['url']})"
            
            f.write(f"| {status_emoji} | {run_link} | {run['user']} | {created} | {run['state']} |\n")

        # Usage Instructions
        f.write("\n## 🔧 Usage Instructions\n\n")
        f.write("### Available Commands\n")
        f.write("- `/runs_report` - Generate this report for the default project\n")
        f.write("- `/runs_report <entity> <project>` - Generate report for specific project\n")
        f.write("- `/runs_report <entity> <project> <limit>` - Generate report with custom limit\n\n")
        f.write("### Filtering Runs\n")
        f.write("You can filter runs by:\n")
        f.write("- State: `finished`, `running`, `failed`, `crashed`\n")
        f.write("- Tags: Add tags to your runs for better organization\n")
        f.write("- Date range: Filter by creation or update date\n\n")

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
