import tensorflow as tf
import numpy as np
import os
import logging
from keras._tf_keras.keras.saving import load_model
from keras._tf_keras.keras.utils import load_img, img_to_array
from keras._tf_keras.keras.ops import expand_dims, sigmoid


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(image_path):
    """Preprocesses a single image for prediction."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1)
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
    model = load_model(model_path)
    image_paths = get_random_images(base_dir, n)
    results = []

    for img_path in image_paths:
        identifier = os.path.basename(img_path).split('_')[0]  # Extract ID
        
        # Load image and preprocess
        img = load_img(img_path, target_size=(224, 224), color_mode="grayscale")
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)  # Create batch axis
        
        # Get prediction (output is a probability distribution over the classes)
        predictions = model.predict(img_array)
        score = float(sigmoid(predictions[0][0]))
        
        # Get the predicted class index (argmax to get the highest probability)
        predicted_class_idx = tf.argmax(predictions, axis=-1).numpy()[0]

        # Get the corresponding class label (if class_names is provided)
        predicted_class = class_names[predicted_class_idx] if class_names else str(predicted_class_idx).zfill(8)

        confidence = np.max(predictions)  # The confidence for the predicted class
        

        result = {
            "image": img_path,
            "identifier": identifier,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "score": score,
        }
        results.append(result)

        logging.info(f"Image: {img_path}, Identifier: {identifier}, Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

    return results
