import streamlit as st
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
from PIL import Image


# LOAD PRECOMPUTED FEATURES

with open('images_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

image_features = np.array(image_features)


# LOAD MODEL

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# STREAMLIT UI

st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
st.title("👗 Complementary Fashion Recommendation System")
st.write("Upload an image to find visually similar fashion items!")

uploaded_file = st.file_uploader("Upload your fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=50)

    
    # Save temporarily
    input_path = "temp_input.jpg"
    input_image.save(input_path)

    # Extract features from uploaded image
    img = image.load_img(input_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    input_features = model.predict(img_array, verbose=0).flatten()

  
    # FIND SIMILAR IMAGES
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(image_features)
    distances, indices = neighbors.kneighbors([input_features])

    st.subheader("5 Recommended Items:")

    cols = st.columns(5)
    for i, col in enumerate(cols):
        idx = indices[0][i+1] if i+1 < len(indices[0]) else indices[0][0]
        col.image(filenames[idx], use_container_width=60, caption=f"Recommendation #{i+1}")
