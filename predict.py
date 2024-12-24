import tensorflow as tf
import numpy as np
import cv2
import os
import random

# Model path
MODEL_SAVE_PATH = "fingerprint_model.h5"

# Define image size
IMG_SIZE = (224, 224)

# Base directory for images
BASE_DIR = './fingerprints/images/latent/png'  # Modify this to your actual directory

# Load the saved model
print("Loading the model...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print("Model loaded successfully!")

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, IMG_SIZE)  # Resize to the expected input size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to get N random images from the dataset
def get_random_images(base_dir, n=5):
    all_images = []
    for label_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, label_folder)
        if os.path.isdir(folder_path):
            all_images += [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.png')]
    
    # Randomly sample N images
    random_images = random.sample(all_images, min(n, len(all_images)))
    print(f"Selected {len(random_images)} random images for prediction.")
    return random_images

# Prediction function for multiple images
def predict_multiple(base_dir, n=5):
    # Get random images
    random_images = get_random_images(base_dir, n)
    
    results = []
    
    for image_path in random_images:
        # Preprocess the image
        img = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img)
        confidence = prediction[0][0]  # Sigmoid output (0 to 1)
        result = "Likely a criminal" if confidence > 0.5 else "Not a criminal"
        
        # Store the result
        results.append({
            "image_path": image_path,
            "prediction": result,
            "confidence": confidence
        })
    
    # Print all results
    for res in results:
        print(f"Image: {res['image_path']} -> Prediction: {res['prediction']} (Confidence: {res['confidence']:.2f})")
    
    return results

# Test the prediction function with N random images
N = 5  # Adjust the number of random images to test
results = predict_multiple(BASE_DIR, n=N)
