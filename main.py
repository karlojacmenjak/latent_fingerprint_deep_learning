import threading

import tensorflow as tf
from train_and_save import train_and_save_model
from predict_from_model import predict_multiple

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for GPUs to dynamically allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

MODEL_PATH = "fingerprint_model.h5"
BASE_DIR = "./fingerprints/images/latent/png"
SUBSET_LIMIT = 100

def train_model_thread():
    train_and_save_model(BASE_DIR, MODEL_PATH, subset_limit=SUBSET_LIMIT)

def predict_thread():
    predict_multiple(BASE_DIR, MODEL_PATH, n=5)

if __name__ == "__main__":
    t_thread = threading.Thread(target=train_model_thread)
    p_thread = threading.Thread(target=predict_thread)

    t_thread.start()
    t_thread.join()  # Wait for training to complete before predicting

    p_thread.start()
    p_thread.join()
