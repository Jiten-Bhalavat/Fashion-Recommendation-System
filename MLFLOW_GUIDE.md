# MLflow Integration Guide for Fashion Recommender System

## 📊 Overview

This Fashion Recommender System now includes comprehensive MLflow tracking to monitor experiments, parameters, metrics, and models.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Feature Extraction with MLflow

```bash
python train_with_mlflow.py
```

This will:
- Extract features from fashion images using ResNet50
- Log all parameters, metrics, and artifacts to MLflow
- Save embeddings as pickle files
- Track processing time and feature statistics

### 3. Run Streamlit App with MLflow

```bash
streamlit run main_with_mlflow.py
```

This will:
- Launch the fashion recommendation interface
- Track each recommendation request
- Log processing times and similarity metrics
- Store uploaded images and recommendations

### 4. View MLflow UI

```bash
mlflow ui
```

Then open: [http://localhost:5000](http://localhost:5000)

## 📁 Project Structure

```
Fashion-Recommender-system/
├── train_with_mlflow.py      # Feature extraction with MLflow tracking
├── main_with_mlflow.py        # Streamlit app with MLflow tracking
├── mlflow_config.py           # MLflow configuration
├── view_experiments.py        # CLI tool to view experiments
├── mlruns/                    # MLflow tracking data (auto-generated)
├── app.py                     # Original feature extraction script
├── main.py                    # Original Streamlit app (no MLflow)
└── test.py                    # Test script
```

## 🔧 MLflow Configuration

### Experiments

Three experiments are configured:

1. **Fashion-Recommender-Feature-Extraction**
   - Tracks feature extraction runs
   - Logs model parameters and build time
   - Tracks processing metrics

2. **Fashion-Recommender-Inference**
   - Tracks recommendation requests
   - Logs similarity scores and processing times
   - Stores uploaded images and results

3. **Fashion-Recommender-Evaluation**
   - Reserved for model evaluation
   - Performance benchmarking

### Tracked Parameters

#### Feature Extraction
- `model_name`: ResNet50
- `weights`: imagenet
- `input_shape`: (224, 224, 3)
- `pooling_layer`: GlobalMaxPooling2D
- `n_neighbors`: Number of neighbors for KNN
- `knn_algorithm`: brute
- `distance_metric`: euclidean
- `max_images`: Number of images to process
- `normalize_features`: True/False

#### Inference
- `n_neighbors`: Number of recommendations
- `knn_algorithm`: Algorithm used
- `distance_metric`: Distance metric
- `uploaded_file_name`: Name of uploaded image
- `uploaded_file_size_kb`: File size

### Tracked Metrics

#### Feature Extraction
- `model_build_time_sec`: Time to build model
- `total_model_params`: Number of model parameters
- `feature_extraction_time_sec`: Total extraction time
- `avg_time_per_image_sec`: Average per image
- `failed_images`: Number of failed extractions
- `successful_extractions`: Number of successful extractions
- `feature_dimension`: Dimension of feature vectors
- `features_mean`, `features_std`, `features_min`, `features_max`: Feature statistics

#### Inference
- `feature_extraction_time_sec`: Time to extract features from query
- `recommendation_time_sec`: Time to find neighbors
- `total_processing_time_sec`: Total processing time
- `avg_distance_to_neighbors`: Average distance
- `min_distance`, `max_distance`: Distance statistics
- `n_recommendations_shown`: Number of recommendations

### Logged Artifacts

- **Feature Extraction**: 
  - `image_features_embedding.pkl`
  - `img_files.pkl`
  - Model (Keras format)

- **Inference**:
  - Uploaded images
  - Recommendations CSV file

## 📊 Viewing Experiments

### Using MLflow UI

```bash
mlflow ui
```

Features:
- Compare runs side-by-side
- Visualize metrics over time
- Download artifacts
- Search and filter runs

### Using CLI Tool

```bash
# List all experiments
python view_experiments.py --list

# View runs for specific experiment
python view_experiments.py --experiment "Fashion-Recommender-Feature-Extraction" --runs 10

# Compare runs by metric
python view_experiments.py --experiment "Fashion-Recommender-Feature-Extraction" --compare "feature_extraction_time_sec" --top 5

# View specific run details
python view_experiments.py --run-id <run_id>
```

## 🎯 Common Use Cases

### 1. Track Different Model Configurations

Modify `mlflow_config.py` to test different parameters:

```python
CONFIG = {
    "model_name": "ResNet50",  # Try: VGG16, InceptionV3
    "n_neighbors": 10,         # Try: 5, 10, 15
    "distance_metric": "euclidean"  # Try: cosine, manhattan
}
```

### 2. Compare Processing Times

```bash
python view_experiments.py --experiment "Fashion-Recommender-Feature-Extraction" --compare "feature_extraction_time_sec"
```

### 3. Analyze Recommendation Quality

Check metrics in MLflow UI:
- Average distance to neighbors
- Similarity scores
- User feedback (if implemented)

### 4. Export Best Model

```python
import mlflow

# Get best run
runs = mlflow.search_runs(
    experiment_names=["Fashion-Recommender-Feature-Extraction"],
    order_by=["metrics.feature_extraction_time_sec ASC"],
    max_results=1
)

best_run_id = runs.iloc[0].run_id

# Load model
model = mlflow.keras.load_model(f"runs:/{best_run_id}/resnet50_feature_extractor")
```

## 🔐 Remote Tracking Server (Optional)

To use a remote MLflow tracking server:

1. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Update `mlflow_config.py`:
```python
MLFLOW_TRACKING_URI = "http://your-server:5000"
```

## 📈 Best Practices

1. **Always log parameters**: Include all hyperparameters
2. **Log metrics consistently**: Use same metric names across runs
3. **Tag experiments**: Add meaningful tags for filtering
4. **Version datasets**: Log dataset version or hash
5. **Document runs**: Add notes about significant findings
6. **Clean up old runs**: Periodically archive completed experiments

## 🐛 Troubleshooting

### Issue: MLflow UI not starting
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Use different port
mlflow ui --port 5001
```

### Issue: Runs not appearing
```bash
# Verify tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Check experiments
python view_experiments.py --list
```

### Issue: Artifacts not loading
- Check `mlruns/` directory permissions
- Verify artifact paths in MLflow UI
- Check disk space

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

## 🤝 Contributing

To add new metrics or parameters:

1. Update `mlflow_config.py` with new configurations
2. Add logging calls in training/inference scripts
3. Update this guide with new tracked values
4. Test with `view_experiments.py`

---

**Happy Experimenting! 🚀**

