"""
Fashion Recommender - Streamlit App with MLflow Tracking
This app provides fashion recommendations and logs user interactions with MLflow
"""

import streamlit as st
import tensorflow
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
import mlflow
import mlflow.tensorflow
import time
from datetime import datetime

# MLflow Configuration
EXPERIMENT_NAME = "Fashion-Recommender-Inference"
mlflow.set_experiment(EXPERIMENT_NAME)

# Configuration
CONFIG = {
    "n_neighbors": 6,
    "knn_algorithm": "brute",
    "distance_metric": "euclidean",
    "model_name": "ResNet50",
    "input_shape": (224, 224, 3),
}

# Load pre-computed features
@st.cache_resource
def load_features():
    """Load pre-computed image features and file list"""
    features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
    img_files_list = pickle.load(open("img_files.pkl", "rb"))
    return features_list, img_files_list

@st.cache_resource
def load_model():
    """Load ResNet50 model for feature extraction"""
    model = ResNet50(weights="imagenet", include_top=False, input_shape=CONFIG["input_shape"])
    model.trainable = False
    model = Sequential([model, GlobalMaxPooling2D()])
    return model

# Load resources
features_list, img_files_list = load_features()
model = load_model()

# Streamlit UI
st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")

st.title('👗 Fashion Recommender System')
st.markdown("### Upload an image to find similar fashion items")

# Sidebar for MLflow tracking info
with st.sidebar:
    st.header("📊 MLflow Tracking")
    st.info("""
    This app logs recommendations to MLflow.
    
    **Tracked Metrics:**
    - Feature extraction time
    - Recommendation time
    - Average distance to neighbors
    - Number of recommendations
    
    To view logs, run:
    ```bash
    mlflow ui
    ```
    Then visit: http://localhost:5000
    """)
    
    enable_mlflow = st.checkbox("Enable MLflow Tracking", value=True)
    
    st.header("⚙️ Settings")
    n_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    CONFIG["n_neighbors"] = n_recommendations + 1  # +1 because first result is the query image itself

def save_file(uploaded_file):
    """Save uploaded file to disk"""
    try:
        os.makedirs("uploader", exist_ok=True)
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return 0

def extract_img_features(img_path, model):
    """Extract normalized features from uploaded image"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img, verbose=0)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)
    
    return result_normlized

def recommend(features, features_list, n_neighbors):
    """Find similar images using KNN"""
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=CONFIG["knn_algorithm"],
        metric=CONFIG["distance_metric"]
    )
    neighbors.fit(features_list)
    
    distances, indices = neighbors.kneighbors([features])
    
    return indices, distances

# File uploader
uploaded_file = st.file_uploader("Choose your image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    
    # Start MLflow run if enabled
    if enable_mlflow:
        mlflow.start_run(run_name=f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "n_neighbors": CONFIG["n_neighbors"],
            "knn_algorithm": CONFIG["knn_algorithm"],
            "distance_metric": CONFIG["distance_metric"],
            "uploaded_file_name": uploaded_file.name,
            "uploaded_file_size_kb": uploaded_file.size / 1024
        })
    
    try:
        if save_file(uploaded_file):
            col_input, col_space = st.columns([1, 2])
            
            with col_input:
                # Display uploaded image
                st.subheader("📸 Your Image")
                show_images = Image.open(uploaded_file)
                size = (300, 300)
                resized_im = show_images.resize(size)
                st.image(resized_im, use_container_width=True)
            
            # Extract features
            st.markdown("---")
            st.subheader("🔍 Finding Similar Items...")
            
            feature_start = time.time()
            features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
            feature_time = time.time() - feature_start
            
            if enable_mlflow:
                mlflow.log_metric("feature_extraction_time_sec", feature_time)
            
            # Get recommendations
            rec_start = time.time()
            img_indices, distances = recommend(features, features_list, CONFIG["n_neighbors"])
            rec_time = time.time() - rec_start
            
            if enable_mlflow:
                mlflow.log_metric("recommendation_time_sec", rec_time)
                mlflow.log_metric("total_processing_time_sec", feature_time + rec_time)
                mlflow.log_metric("avg_distance_to_neighbors", float(np.mean(distances[0][1:])))
                mlflow.log_metric("min_distance", float(np.min(distances[0][1:])))
                mlflow.log_metric("max_distance", float(np.max(distances[0][1:])))
                mlflow.log_metric("n_recommendations_shown", n_recommendations)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Feature Extraction", f"{feature_time:.3f}s")
            with metric_col2:
                st.metric("Recommendation", f"{rec_time:.3f}s")
            with metric_col3:
                st.metric("Total Time", f"{feature_time + rec_time:.3f}s")
            
            st.markdown("---")
            st.subheader("✨ Recommended Items")
            
            # Display recommendations in columns
            cols = st.columns(n_recommendations)
            
            for i in range(n_recommendations):
                with cols[i]:
                    idx = img_indices[0][i+1]  # Skip first result (query image itself)
                    distance = distances[0][i+1]
                    
                    st.image(img_files_list[idx], use_container_width=True)
                    st.caption(f"Match: {(1-distance)*100:.1f}%")
                    st.caption(f"Distance: {distance:.4f}")
            
            # Log artifacts
            if enable_mlflow:
                # Save uploaded image as artifact
                mlflow.log_artifact(os.path.join("uploader", uploaded_file.name), "uploaded_images")
                
                # Create and log recommendations summary
                recommendations_df = pd.DataFrame({
                    'rank': range(1, n_recommendations + 1),
                    'image_path': [img_files_list[img_indices[0][i+1]] for i in range(n_recommendations)],
                    'distance': [distances[0][i+1] for i in range(n_recommendations)],
                    'similarity_score': [(1-distances[0][i+1]) * 100 for i in range(n_recommendations)]
                })
                
                recommendations_df.to_csv("temp_recommendations.csv", index=False)
                mlflow.log_artifact("temp_recommendations.csv", "recommendations")
                os.remove("temp_recommendations.csv")
                
                mlflow.end_run()
            
            st.success(f"✅ Found {n_recommendations} similar items!")
            
            if enable_mlflow:
                st.info("📊 Recommendation logged to MLflow")
        
        else:
            st.error("❌ Error occurred while saving the file")
            if enable_mlflow:
                mlflow.end_run(status="FAILED")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        if enable_mlflow and mlflow.active_run():
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")

else:
    st.info("👆 Please upload an image to get started")
    
    # Show example images
    st.markdown("---")
    st.subheader("📋 Example Images")
    st.markdown("You can try with these sample images:")
    
    if os.path.exists("sample"):
        sample_images = [f for f in os.listdir("sample") if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        cols = st.columns(len(sample_images))
        
        for i, img_name in enumerate(sample_images):
            with cols[i]:
                st.image(os.path.join("sample", img_name), caption=img_name, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, TensorFlow, and MLflow</p>
    <p>Model: ResNet50 | Algorithm: K-Nearest Neighbors</p>
</div>
""", unsafe_allow_html=True)

