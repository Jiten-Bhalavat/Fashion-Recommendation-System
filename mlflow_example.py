"""
Simple MLflow Example
Demonstrates basic MLflow tracking for the Fashion Recommender
"""

import mlflow
from mlflow_config import setup_experiment, log_system_info
import time
import numpy as np

def example_experiment():
    """Run a simple example experiment with MLflow tracking"""
    
    # Setup experiment
    setup_experiment("feature_extraction")
    
    # Start an MLflow run
    with mlflow.start_run(run_name="example_run"):
        
        print("\n" + "="*60)
        print("Running Example MLflow Experiment")
        print("="*60)
        
        # Log system information
        log_system_info()
        
        # Example parameters
        params = {
            "model": "ResNet50",
            "n_neighbors": 6,
            "distance_metric": "euclidean",
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        print("\n[Logging parameters...]")
        mlflow.log_params(params)
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Simulate some processing
        print("\n[Simulating feature extraction...]")
        start_time = time.time()
        time.sleep(2)  # Simulate work
        processing_time = time.time() - start_time
        
        # Example metrics
        metrics = {
            "processing_time_sec": processing_time,
            "feature_dimension": 2048,
            "accuracy": 0.95,
            "avg_similarity": 0.87,
            "images_processed": 500
        }
        
        print("\n[Logging metrics...]")
        mlflow.log_metrics(metrics)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Log tags
        mlflow.set_tags({
            "environment": "development",
            "dataset": "fashion_small",
            "purpose": "example"
        })
        
        # Create and log a sample artifact
        print("\n[Creating sample artifact...]")
        sample_data = np.random.rand(100, 2048)
        artifact_path = "sample_features.npy"
        np.save(artifact_path, sample_data)
        mlflow.log_artifact(artifact_path)
        
        # Clean up
        import os
        os.remove(artifact_path)
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*60)
        print("[SUCCESS] Experiment completed successfully!")
        print("="*60)
        print(f"Run ID: {run_id}")
        print(f"\nView in MLflow UI:")
        print("  1. Run: mlflow ui")
        print("  2. Open: http://localhost:5000")
        print(f"  3. Navigate to run: {run_id[:8]}...")
        print("="*60)

if __name__ == "__main__":
    try:
        example_experiment()
    except Exception as e:
        print(f"[ERROR] {str(e)}")

