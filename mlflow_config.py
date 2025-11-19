"""
MLflow Configuration for Fashion Recommender System
Central configuration for all MLflow experiments
"""

import mlflow
import os

# MLflow Tracking URI - Change this to use remote tracking server
# MLFLOW_TRACKING_URI = "http://localhost:5000"  # Local server
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"    # SQLite backend
MLFLOW_TRACKING_URI = "./mlruns"  # Local file store (default)

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment configurations
EXPERIMENTS = {
    "feature_extraction": {
        "name": "Fashion-Recommender-Feature-Extraction",
        "description": "Feature extraction from fashion images using ResNet50",
        "tags": {
            "project": "fashion-recommender",
            "stage": "training",
            "model": "resnet50"
        }
    },
    "inference": {
        "name": "Fashion-Recommender-Inference",
        "description": "Real-time fashion recommendations",
        "tags": {
            "project": "fashion-recommender",
            "stage": "inference",
            "model": "knn"
        }
    },
    "evaluation": {
        "name": "Fashion-Recommender-Evaluation",
        "description": "Model evaluation and performance metrics",
        "tags": {
            "project": "fashion-recommender",
            "stage": "evaluation"
        }
    }
}

# Default parameters
DEFAULT_PARAMS = {
    "model_name": "ResNet50",
    "weights": "imagenet",
    "input_shape": (224, 224, 3),
    "n_neighbors": 6,
    "knn_algorithm": "brute",
    "distance_metric": "euclidean",
    "normalize_features": True
}

def setup_experiment(experiment_type="feature_extraction"):
    """
    Setup MLflow experiment with proper configuration
    
    Args:
        experiment_type: One of 'feature_extraction', 'inference', 'evaluation'
    
    Returns:
        experiment: MLflow experiment object
    """
    if experiment_type not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    config = EXPERIMENTS[experiment_type]
    
    # Create or get experiment
    experiment = mlflow.set_experiment(config["name"])
    
    # Set experiment description if not already set
    if experiment.tags.get("mlflow.note.content") != config["description"]:
        mlflow.set_experiment_tag("mlflow.note.content", config["description"])
    
    # Set experiment tags
    for key, value in config["tags"].items():
        mlflow.set_experiment_tag(key, value)
    
    print(f"[OK] MLflow experiment '{config['name']}' is ready")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  Experiment ID: {experiment.experiment_id}")
    
    return experiment

def log_system_info():
    """Log system and environment information"""
    import platform
    import sklearn
    import numpy as np
    
    mlflow.log_param("python_version", platform.python_version())
    try:
        import tensorflow as tf
        mlflow.log_param("tensorflow_version", tf.__version__)
    except ImportError:
        mlflow.log_param("tensorflow_version", "not_installed")
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("numpy_version", np.__version__)
    mlflow.log_param("platform", platform.platform())
    mlflow.log_param("processor", platform.processor())

def get_latest_run(experiment_name):
    """Get the latest run from an experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        return None
    
    return runs.iloc[0]

def compare_runs(experiment_name, metric_name="feature_extraction_time_sec", top_n=5):
    """Compare top N runs by a specific metric"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} ASC"],
        max_results=top_n
    )
    
    return runs

if __name__ == "__main__":
    print("="*60)
    print("MLflow Configuration for Fashion Recommender System")
    print("="*60)
    print(f"\nTracking URI: {MLFLOW_TRACKING_URI}")
    print(f"\nConfigured Experiments:")
    for exp_type, config in EXPERIMENTS.items():
        print(f"\n  {exp_type}:")
        print(f"    Name: {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Tags: {config['tags']}")
    
    print("\n" + "="*60)
    print("Setup complete! Use setup_experiment() to initialize.")
    print("="*60)

