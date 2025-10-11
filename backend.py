import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

DATASET_DIR = r"Images"


MAX_IMAGES = 30000   

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


all_images = []
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(root, file))

if MAX_IMAGES:
    all_images = all_images[:MAX_IMAGES]

print(f"Total images to process: {len(all_images)}")


features_list = []
valid_filenames = []

for img_path in tqdm(all_images, desc="Extracting features"):
    feat = extract_features(img_path)
    if feat is not None:
        features_list.append(feat)
        valid_filenames.append(img_path)


features_array = np.array(features_list)
print(f"\n Feature extraction complete! Shape: {features_array.shape}")


with open('images_features.pkl', 'wb') as f:
    pickle.dump(features_array, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(valid_filenames, f)

print("\n Features and filenames saved successfully!")
print(" images_features.pkl and filenames.pkl created.")
