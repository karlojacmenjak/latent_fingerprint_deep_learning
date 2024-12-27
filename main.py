import tensorflow as tf
from train_and_save import train_and_save_model
from predict_from_model import predict_multiple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"Failed to set GPU memory growth: {e}")

# Paths and settings
MODEL_PATH = "fingerprint_model.h5"
BASE_DIR = "./fingerprints/images/latent/png"
SUBSET_LIMIT = 1000
PREDICT_COUNT = 5

if __name__ == "__main__":
    logging.info("Starting training...")
    train_and_save_model(BASE_DIR, MODEL_PATH, subset_limit=SUBSET_LIMIT)

    logging.info("Starting prediction...")
    results = predict_multiple(BASE_DIR, MODEL_PATH, n=PREDICT_COUNT)
    for result in results:
        logging.info(f"Summary -> ID: {result['identifier']}, Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.2f}")
