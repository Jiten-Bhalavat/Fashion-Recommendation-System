"""
MLflow Experiment Viewer
View and compare MLflow experiments from the command line
"""

import mlflow
import pandas as pd
import argparse
from mlflow_config import EXPERIMENTS, MLFLOW_TRACKING_URI
from tabulate import tabulate

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def list_experiments():
    """List all experiments"""
    experiments = mlflow.search_experiments()
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    print("\n" + "="*80)
    print("MLFLOW EXPERIMENTS")
    print("="*80)
    
    exp_data = []
    for exp in experiments:
        exp_data.append([
            exp.experiment_id,
            exp.name,
            exp.lifecycle_stage,
            mlflow.search_runs([exp.experiment_id]).shape[0] if exp.lifecycle_stage == 'active' else 0
        ])
    
    print(tabulate(exp_data, headers=["ID", "Name", "Status", "Runs"], tablefmt="grid"))
    print()

def view_runs(experiment_name, top_n=10):
    """View runs for a specific experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found!")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=top_n
    )
    
    if len(runs) == 0:
        print(f"No runs found for experiment '{experiment_name}'")
        return
    
    print("\n" + "="*80)
    print(f"RUNS FOR: {experiment_name}")
    print("="*80)
    
    # Select relevant columns
    display_cols = ['run_id', 'start_time', 'status']
    
    # Add metric columns if they exist
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    
    # Display summary
    display_data = runs[display_cols + metric_cols[:5] + param_cols[:3]].head(top_n)
    
    # Shorten run_id for display
    display_data['run_id'] = display_data['run_id'].str[:8]
    
    print(display_data.to_string(index=False))
    print(f"\nShowing {min(top_n, len(runs))} of {len(runs)} runs")
    print()

def compare_runs(experiment_name, metric_name, top_n=5):
    """Compare runs by a specific metric"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found!")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} ASC"],
        max_results=top_n
    )
    
    if len(runs) == 0:
        print(f"No runs found for experiment '{experiment_name}'")
        return
    
    print("\n" + "="*80)
    print(f"TOP {top_n} RUNS BY {metric_name}")
    print(f"Experiment: {experiment_name}")
    print("="*80)
    
    # Select columns
    display_cols = ['run_id', 'start_time', f'metrics.{metric_name}']
    
    # Add related metrics
    related_metrics = [col for col in runs.columns if col.startswith('metrics.') and col != f'metrics.{metric_name}']
    display_data = runs[display_cols + related_metrics[:3]].head(top_n)
    
    # Shorten run_id
    display_data['run_id'] = display_data['run_id'].str[:8]
    
    print(display_data.to_string(index=False))
    print()

def get_run_details(run_id):
    """Get detailed information about a specific run"""
    run = mlflow.get_run(run_id)
    
    print("\n" + "="*80)
    print(f"RUN DETAILS: {run_id}")
    print("="*80)
    
    print(f"\nExperiment: {mlflow.get_experiment(run.info.experiment_id).name}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {run.info.start_time}")
    print(f"End Time: {run.info.end_time}")
    print(f"Duration: {(run.info.end_time - run.info.start_time) / 1000:.2f}s")
    
    print("\n" + "-"*40)
    print("PARAMETERS")
    print("-"*40)
    for key, value in run.data.params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*40)
    print("METRICS")
    print("-"*40)
    for key, value in run.data.metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*40)
    print("ARTIFACTS")
    print("-"*40)
    artifacts = mlflow.artifacts.list_artifacts(run_id)
    for artifact in artifacts:
        print(f"  {artifact.path}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="View MLflow experiments")
    parser.add_argument("--list", "-l", action="store_true", help="List all experiments")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment name to view")
    parser.add_argument("--runs", "-r", type=int, default=10, help="Number of runs to show")
    parser.add_argument("--compare", "-c", type=str, help="Metric name to compare runs")
    parser.add_argument("--run-id", type=str, help="Specific run ID to view details")
    parser.add_argument("--top", "-t", type=int, default=5, help="Top N runs for comparison")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.run_id:
        get_run_details(args.run_id)
    elif args.experiment:
        if args.compare:
            compare_runs(args.experiment, args.compare, args.top)
        else:
            view_runs(args.experiment, args.runs)
    else:
        # Default: show all experiments and recent runs
        list_experiments()
        print("\nRecent runs from all experiments:")
        print("-"*80)
        
        for exp_name in [exp["name"] for exp in EXPERIMENTS.values()]:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=3
                )
                if len(runs) > 0:
                    print(f"\n{exp_name}: {len(runs)} recent runs")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")

