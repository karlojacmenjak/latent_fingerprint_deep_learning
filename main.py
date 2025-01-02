import tensorflow as tf
from constants import BASE_DIR, LABEL_PATH, MODEL_PATH, PREDICT_COUNT, SUBSET_LIMIT
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


if __name__ == "__main__":
    logging.info("Starting training...")
    train_and_save_model(BASE_DIR, MODEL_PATH, subset_limit=SUBSET_LIMIT)

    logging.info("Starting prediction...")
    results = predict_multiple(BASE_DIR, MODEL_PATH, n=PREDICT_COUNT, label_dir=LABEL_PATH)
    for result in results:
        output = ""
        for key, value in result.items():
            output += key.upper() + ": " + str(value) + "\t"
        output += "\n"
        print(output)