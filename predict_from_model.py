import tensorflow as tf
import numpy as np
import os
import cv2
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return np.expand_dims(img / 255.0, axis=0)

def get_random_images(base_dir, n=5):
    image_paths = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.endswith('.png'):
                    image_paths.append(os.path.join(folder_path, img))
    return random.sample(image_paths, min(n, len(image_paths)))

def predict_multiple(base_dir, model_path, n=5):
    model = tf.keras.models.load_model(model_path)
    images = get_random_images(base_dir, n)
    results = []

    for img_path in images:
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        confidence = prediction[0][0]
        result = "Likely a criminal" if confidence > 0.5 else "Not a criminal"
        results.append({"image": img_path, "prediction": result, "confidence": confidence})
        logging.info(f"{img_path}: {result} (Confidence: {confidence:.2f})")

    return results
