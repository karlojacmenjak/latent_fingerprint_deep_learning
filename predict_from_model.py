import tensorflow as tf
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(image_path):
    """Preprocesses a single image for prediction."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0, 1]
    return img

def get_random_images(base_dir, n=5):
    """Gets N random images from the dataset."""
    image_paths = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.endswith('.png'):
                    image_paths.append(os.path.join(folder_path, img))
    return np.random.choice(image_paths, size=min(n, len(image_paths)), replace=False)

def predict_multiple(base_dir, model_path, n=5, class_names=None):
    """Predicts for multiple random images and logs the results."""
    model = tf.keras.models.load_model(model_path)
    image_paths = get_random_images(base_dir, n)
    results = []

    for img_path in image_paths:
        identifier = os.path.basename(img_path).split('_')[0]  # Extract ID
        img = preprocess_image(img_path)  # Preprocess the image
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
        
        prediction = model.predict(img)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        confidence = tf.reduce_max(prediction).numpy()
        
        result = {
            "image": img_path,
            "identifier": identifier,
            "predicted_class": class_names[predicted_class] if class_names else predicted_class,
            "confidence": confidence
        }
        results.append(result)

        logging.info(f"Image: {img_path}, Identifier: {identifier}, Predicted Class: {result['predicted_class']}, Confidence: {confidence:.2f}")

    return results
