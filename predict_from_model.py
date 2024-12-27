import tensorflow as tf
import numpy as np
import os
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(image_path, image_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, image_size) / 255.0
    return tf.expand_dims(img, axis=0)

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
        identifier = os.path.basename(img_path).split('_')[0]
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        predicted_class = tf.argmax(prediction[0]).numpy()
        confidence = tf.reduce_max(prediction[0]).numpy()

        results.append({
            "image": img_path,
            "identifier": identifier,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

        logging.info(f"Image: {img_path}, Identifier: {identifier}, Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    return results
