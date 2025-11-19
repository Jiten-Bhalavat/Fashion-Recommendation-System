
# Fashion Recommender system

With an increase in the standard of living, peoples' attention gradually moved towards fashion that is concerned to be a popular aesthetic expression. Humans are inevitably drawn towards something that is visually more attractive. This tendency of humans has led to the development of the fashion industry over the course of time. However, given too many options of garments on the e-commerce websites, has presented new challenges to the customers in identifying their correct outfit. Thus, in this project, we proposed a personalized Fashion Recommender system that generates recommendations for the user based on an input given. Unlike the conventional systems that rely on the user's previous purchases and history, this project aims at using an image of a product given as input by the user to generate recommendations since many-a-time people see something that they are interested in and tend to look for products that are similar to that. We use neural networks to process the images from Fashion Product Images Dataset and the Nearest neighbour backed recommender to generate the final recommendations.

## Introduction

Humans are inevitably drawn towards something that is visually more attractive. This tendency of 
humans has led to development of fashion industry over the course of time. With introduction of 
recommender systems in multiple domains, retail industries are coming forward with investments in 
latest technology to improve their business. Fashion has been in existence since centuries and will be 
prevalent in the coming days as well. Women are more correlated with fashion and style, and they 
have a larger product base to deal with making it difficult to take decisions. It has become an important 
aspect of life for modern families since a person is more often than not judged based on his attire. 
Moreover, apparel providers need their customers to explore their entire product line so they can 
choose what they like the most which is not possible by simply going into a cloth store.

## Related work

In the online internet era, the idea of Recommendation technology was initially introduced in the mid-90s. Proposed CRESA that combined visual features, textual attributes and visual attention of 
the user to build the clothes profile and generate recommendations. Utilized fashion magazines 
photographs to generate recommendations. Multiple features from the images were extracted to learn 
the contents like fabric, collar, sleeves, etc., to produce recommendations. In order to meet the 
diverse needs of different users, an intelligent Fashion recommender system is studied based on 
the principles of fashion and aesthetics. To generate garment recommendations, customer ratings and 
clothing were utilized in The history of clothes and accessories, weather conditions were 
considered in to generate recommendations.

##  Proposed methodology

In this project, we propose a model that uses Convolutional Neural Network and the Nearest 
neighbour backed recommender. As shown in the figure Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in 
inventory. The nearest neighbour's algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

![Work Model](Demo/work-model.png)

## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![ResNet50 Architecture](Demo/resnet.png)

## Getting the inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![Inventory Sample](Demo/inventry.png)

## Recommendation generation

To generate recommendations, our proposed approach uses Sklearn Nearest neighbours Oh Yeah. This allows us to find the nearest neighbours for the 
given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.

## Experiment and results

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. 
Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441
garment images. The networks are trained and validated on the dataset taken. The training results 
show a great accuracy of the model with low error, loss and good f-score.

### Dataset Link

[Kaggle Dataset Big size 15 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

## Screenshots

### Simple App UI

![App UI](Demo/2021-11-25.png)

### Outfits generated by our approach for the given input image

![Recommendation Example 1](Demo/2021-11-25%20(1).png)

![Recommendation Example 2](Demo/2021-11-25%20(4).png)

![Recommendation Example 3](Demo/2021-11-25%20(3).png)

## 🚀 Quick Start (Demo with 500 Images)

This repository includes pre-processed data for 500 images, so you can run the app immediately!

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Fashion-Recommender-system.git
cd Fashion-Recommender-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run main.py
```

That's it! The app will use the included `image_features_embedding.pkl` and `img_files.pkl` files with 500 pre-processed images.

---

## 📦 Running with Full Dataset (44,441 Images)

If you want to use the complete dataset for better recommendations, follow these steps:

### Step 1: Download the Dataset

Download the Fashion Product Images dataset from Kaggle:
- **Small Dataset (Recommended):** [572 MB - 44,441 images](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)
- **Large Dataset:** [15 GB with metadata](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

### Step 2: Extract the Dataset

Extract the downloaded dataset and place the `images` folder inside `fashion_small/` directory:

```
Fashion-Recommender-system/
├── fashion_small/
│   └── images/
│       ├── 10000.jpg
│       ├── 10001.jpg
│       └── ... (44,441 images)
├── main.py
├── app.py
└── ...
```

### Step 3: Modify `app.py` to Process All Images

Open `app.py` and change line 37-38 from:

```python
img_files = img_files[:500]
print(f"Processing first {len(img_files)} images...")
```

To:

```python
# Process all images (remove the limit)
print(f"Processing all {len(img_files)} images...")
```

### Step 4: Generate Feature Embeddings

Run the feature extraction script (this will take some time depending on your hardware):

```bash
python app.py
```

This will:
- Process all 44,441 images using ResNet50
- Generate `image_features_embedding.pkl` (feature embeddings)
- Generate `img_files.pkl` (image file paths)
- Progress bar will show the processing status

**Note:** Processing all images may take 30-60 minutes depending on your CPU/GPU.

### Step 5: Run the Application

```bash
streamlit run main.py
```

Now your recommender system will use the complete dataset for more accurate recommendations!

---

## 📁 Project Structure

```
Fashion-Recommender-system/
├── main.py                          # Streamlit web application
├── app.py                           # Feature extraction script
├── test.py                          # Testing script with OpenCV
├── requirements.txt                 # Python dependencies
├── image_features_embedding.pkl     # Pre-computed feature embeddings (500 images)
├── img_files.pkl                    # Image file paths (500 images)
├── fashion_small/
│   └── images/                      # Fashion dataset images (500 included, 44,441 for full)
├── sample/                          # Sample test images
├── uploader/                        # Temporary upload directory
└── Demo/                            # Screenshots and documentation images
```

---

## 💻 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- TensorFlow 2.7.0
- Streamlit 1.2.0
- NumPy 1.21.4
- Pandas 1.3.4
- Pillow 8.4.0
- scikit-learn 1.0.1
- OpenCV 4.5.4
- tqdm 4.62.3

---

## 🎯 Usage

### Using the Web Application

1. Start the Streamlit server:
```bash
streamlit run main.py
```

2. Open your browser (usually opens automatically at `http://localhost:8501`)

3. Upload a fashion item image using the file uploader

4. Get 5 similar fashion recommendations instantly!

### Using the Test Script

To test recommendations with OpenCV visualization:

```bash
python test.py
```

This will show recommendations for the sample image using OpenCV windows.

---

## 🔧 Customization

### Change Number of Recommendations

In `main.py`, modify line 48:

```python
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
```

Change `n_neighbors` to get more/fewer recommendations (first result is the input image itself).

### Use Different Distance Metrics

You can change the similarity metric in `main.py`:

```python
# Options: 'euclidean', 'cosine', 'manhattan'
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
```

---

## ⚠️ Troubleshooting

### Issue: "No module named 'tensorflow'"
```bash
pip install tensorflow==2.7.0
```

### Issue: "Cannot find image_features_embedding.pkl"
Run `python app.py` first to generate the pickle files.

### Issue: "Out of memory" when processing full dataset
Reduce batch size or process images in chunks. Alternatively, use the included 500 images.

---

## Built With

- [OpenCV]() - Open Source Computer Vision and Machine Learning software library
- [Tensorflow]() - TensorFlow is an end-to-end open source platform for machine learning.
- [Tqdm]() - tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.
- [streamlit]() - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.
- [pandas]() - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- [Pillow]() - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
- [scikit-learn]() - Scikit-learn is a free software machine learning library for the Python programming language.
- [opencv-python]() - OpenCV is a huge open-source library for computer vision, machine learning, and image processing.

## Conclusion

In this project, we have presented a novel framework for fashion recommendation that is driven by data, 
visually related and simple effective recommendation systems for generating fashion product images. 
The proposed approach uses a two-stage phase. Initially, our proposed approach extracts the features 
of the image using CNN classifier ie., for instance allowing the customers to upload any random 
fashion image from any E-commerce website and later generating similar images to the uploaded image 
based on the features and texture of the input image. It is imperative that such research goes forward 
to facilitate greater recommendation accuracy and improve the overall experience of fashion 
exploration for direct and indirect consumers alike.
