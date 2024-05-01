import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance

import cv2

# Read an image
img1 = cv2.imread('/Users/avilncurry/Desktop/CsvToJs/plot24.png')
img2 = cv2.imread('/Users/avilncurry/Desktop/CsvToJs/plot25.png')

# Resize the image
img_resized1 = cv2.resize(img1, (224, 224))
img_resized2 = cv2.resize(img2, (224, 224))

# Save the image if needed
cv2.imwrite('/Users/avilncurry/Desktop/CsvToJs/rs24.png', img_resized1)

cv2.imwrite('/Users/avilncurry/Desktop/CsvToJs/rs25.png', img_resized2)

# Load VGG16 model
model = VGG16(weights='imagenet', include_top=False)


# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features


# Paths to images
image_path_1 = '/Users/avilncurry/Desktop/CsvToJs/rs24.png'
image_path_2 = '/Users/avilncurry/Desktop/CsvToJs/rs25.png'

# Extract features
features_1 = extract_features(image_path_1, model)
features_2 = extract_features(image_path_2, model)

# Calculate distance between features
euclidean_dist = distance.euclidean(features_1, features_2)
cosine_dist = distance.cosine(features_1, features_2)

print("Euclidean Distance:", euclidean_dist)
print("Cosine Distance:", cosine_dist)
