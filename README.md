# visual product matcher System

A full-stack application that allows users to upload an image of a product and receive visually similar recommendations using deep learning and nearest neighbor search.



## **Project Overview**

 visual product matcher System helps users find similar fashion items from a pre-existing catalog. Users can upload an image, and the system recommends visually similar items. The backend uses **ResNet50** for feature extraction, while **k-Nearest Neighbors (k-NN)** finds similar images.
Dataset - https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---

## **Folder Structure**

.
- ** .venv/ # Python virtual environment  (too big to upload that's why not in the github repo)
- ** images/ # Folder containing the dataset images  (too big to upload that's why not in the github repo)
- ** uploads/ # Folder to store uploaded images temporarily
- ** 16871.jpg # Example image file
- ** app.py # Flask backend application
- ** filenames.pkl # Pickle file containing image filenames
- ** images_features.pkl # Pickle file containing precomputed image features
- ** recom.py # Script for recommendations (optional/utility)


---

## **Technologies Used**

- **Frontend:** React.js (optional if you have a frontend)  
- **Backend:** Flask, Python  
- **Machine Learning:** scikit-learn (NearestNeighbors)  
- **Deep Learning:** TensorFlow Keras (ResNet50)  
- **Data Handling:** NumPy, Pickle  
- **Image Processing:** PIL  

---
How It Works

Upload an image to the backend (via frontend or API).

Features are extracted using ResNet50.

k-NN finds the top similar images from images_features.pkl.

Recommended images are sent as a JSON response with filenames.

Images can be accessed via the /image/<filename> endpoint.

---

Front-end - Streamlit

-- Usage

Upload an image of a fashion item.

Receive a list of recommended images.

Display recommendations on a frontend or CLI as needed.
