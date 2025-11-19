# MLflow Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Your First Experiment
```bash
# Option A: Feature extraction with tracking
python train_with_mlflow.py

# Option B: Streamlit app with tracking
streamlit run main_with_mlflow.py

# Option C: Simple example
python mlflow_example.py
```

### Step 3: View Results
```bash
mlflow ui
```
Open: http://localhost:5000

## 📊 What Gets Tracked?

### Feature Extraction (`train_with_mlflow.py`)
- **Parameters**: Model architecture, hyperparameters
- **Metrics**: Processing time, feature statistics
- **Artifacts**: Feature embeddings, trained model

### Inference (`main_with_mlflow.py`)
- **Parameters**: Number of neighbors, distance metric
- **Metrics**: Recommendation time, similarity scores
- **Artifacts**: Uploaded images, recommendation results

## 🔍 View Experiments

### Web UI (Recommended)
```bash
mlflow ui
```

### Command Line
```bash
# List all experiments
python view_experiments.py --list

# View specific experiment
python view_experiments.py --experiment "Fashion-Recommender-Feature-Extraction"

# Compare runs by metric
python view_experiments.py --experiment "Fashion-Recommender-Feature-Extraction" --compare "feature_extraction_time_sec"
```

## 📝 Common Commands

```bash
# Start MLflow UI
mlflow ui

# Start on different port
mlflow ui --port 5001

# View help
mlflow --help
python view_experiments.py --help
```

## 🎯 Next Steps

1. ✅ Run an experiment: `python train_with_mlflow.py`
2. ✅ View in MLflow UI: `mlflow ui`
3. ✅ Compare different runs
4. ✅ Export best model
5. ✅ Read full guide: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)

## 💡 Tips

- Enable/disable tracking in Streamlit app using the sidebar toggle
- Each run gets a unique ID for easy reference
- Artifacts are stored in `mlruns/` directory
- All experiments are local by default (can be configured for remote tracking)

## 🆘 Need Help?

- Full documentation: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)
- MLflow docs: https://mlflow.org/docs/latest/index.html
- View example: `python mlflow_example.py`

---

**Happy Tracking! 📈**

