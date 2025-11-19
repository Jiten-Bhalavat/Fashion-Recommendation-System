"""
Fashion Recommender - Feature Extraction with MLflow Tracking
This script extracts features from fashion images and logs the experiment with MLflow
"""

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import mlflow
import mlflow.tensorflow
import mlflow.keras
import time
from datetime import datetime

# MLflow Configuration
EXPERIMENT_NAME = "Fashion-Recommender-Feature-Extraction"
mlflow.set_experiment(EXPERIMENT_NAME)

# Hyperparameters and Configuration
CONFIG = {
    "model_name": "ResNet50",
    "weights": "imagenet",
    "include_top": False,
    "input_shape": (224, 224, 3),
    "pooling_layer": "GlobalMaxPooling2D",
    "n_neighbors": 6,
    "knn_algorithm": "brute",
    "distance_metric": "euclidean",
    "max_images": 500,
    "dataset_path": "fashion_small/images",
    "normalize_features": True
}

def extract_features(img_path, model):
    """Extract normalized features from an image"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img, verbose=0)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)
    
    return result_normlized

def main():
    # Start MLflow run
    with mlflow.start_run(run_name=f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("="*60)
        print("Fashion Recommender - MLflow Tracking Enabled")
        print("="*60)
        
        # Log parameters
        mlflow.log_params(CONFIG)
        
        # Build model
        print("\n[1/4] Building ResNet50 Model...")
        start_time = time.time()
        
        base_model = ResNet50(
            weights=CONFIG["weights"],
            include_top=CONFIG["include_top"],
            input_shape=CONFIG["input_shape"]
        )
        base_model.trainable = False
        
        model = Sequential([base_model, GlobalMaxPooling2D()])
        
        model_build_time = time.time() - start_time
        mlflow.log_metric("model_build_time_sec", model_build_time)
        print(f"✓ Model built in {model_build_time:.2f} seconds")
        
        # Log model summary
        model_params = model.count_params()
        mlflow.log_metric("total_model_params", model_params)
        print(f"✓ Total model parameters: {model_params:,}")
        
        # Collect image files
        print("\n[2/4] Collecting image files...")
        img_files = []
        
        if os.path.exists(CONFIG["dataset_path"]):
            for fashion_images in os.listdir(CONFIG["dataset_path"]):
                images_path = os.path.join(CONFIG["dataset_path"], fashion_images)
                if os.path.isfile(images_path):
                    img_files.append(images_path)
        else:
            print(f"⚠ Warning: Dataset path '{CONFIG['dataset_path']}' not found!")
            print("Using sample images instead...")
            CONFIG["dataset_path"] = "sample"
            mlflow.log_param("dataset_path", "sample")
            for sample_img in os.listdir("sample"):
                if sample_img.endswith(('.jpg', '.jpeg', '.png')):
                    img_files.append(os.path.join("sample", sample_img))
        
        total_images = len(img_files)
        img_files = img_files[:CONFIG["max_images"]]
        images_to_process = len(img_files)
        
        mlflow.log_metric("total_images_available", total_images)
        mlflow.log_metric("images_processed", images_to_process)
        
        print(f"✓ Total images found: {total_images}")
        print(f"✓ Processing first {images_to_process} images...")
        
        # Extract features
        print("\n[3/4] Extracting features...")
        extraction_start = time.time()
        image_features = []
        failed_images = 0
        
        for idx, file_path in enumerate(tqdm(img_files, desc="Processing")):
            try:
                features = extract_features(file_path, model)
                image_features.append(features)
            except Exception as e:
                print(f"\n⚠ Error processing {file_path}: {str(e)}")
                failed_images += 1
        
        extraction_time = time.time() - extraction_start
        
        # Log extraction metrics
        mlflow.log_metric("feature_extraction_time_sec", extraction_time)
        mlflow.log_metric("avg_time_per_image_sec", extraction_time / images_to_process)
        mlflow.log_metric("failed_images", failed_images)
        mlflow.log_metric("successful_extractions", len(image_features))
        
        print(f"\n✓ Feature extraction completed in {extraction_time:.2f} seconds")
        print(f"✓ Average time per image: {extraction_time/images_to_process:.4f} seconds")
        print(f"✓ Failed images: {failed_images}")
        
        # Log feature statistics
        if image_features:
            features_array = np.array(image_features)
            mlflow.log_metric("feature_dimension", features_array.shape[1])
            mlflow.log_metric("features_mean", float(np.mean(features_array)))
            mlflow.log_metric("features_std", float(np.std(features_array)))
            mlflow.log_metric("features_min", float(np.min(features_array)))
            mlflow.log_metric("features_max", float(np.max(features_array)))
            
            print(f"\n✓ Feature dimensions: {features_array.shape}")
            print(f"✓ Feature statistics:")
            print(f"  - Mean: {np.mean(features_array):.6f}")
            print(f"  - Std: {np.std(features_array):.6f}")
            print(f"  - Min: {np.min(features_array):.6f}")
            print(f"  - Max: {np.max(features_array):.6f}")
        
        # Save embeddings
        print("\n[4/4] Saving embeddings...")
        pickle.dump(image_features, open("image_features_embedding.pkl", "wb"))
        pickle.dump(img_files, open("img_files.pkl", "wb"))
        
        # Log artifacts
        mlflow.log_artifact("image_features_embedding.pkl")
        mlflow.log_artifact("img_files.pkl")
        
        # Log the model
        mlflow.keras.log_model(model, "resnet50_feature_extractor")
        
        print("✓ Embeddings saved successfully")
        print(f"✓ Files: image_features_embedding.pkl, img_files.pkl")
        
        # Summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Images processed: {len(image_features)}/{images_to_process}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print("="*60)
        print("\nℹ View results in MLflow UI:")
        print("  Run: mlflow ui")
        print("  Then open: http://localhost:5000")
        print("="*60)

if __name__ == "__main__":
    main()

