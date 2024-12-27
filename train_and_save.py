import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_images_with_labels(base_dir, subset_limit=None):
    image_paths = []
    labels = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.endswith('.png'):
                    image_paths.append(os.path.join(folder_path, img))
                    labels.append(folder)  # Folder name as label (class ID)
    if subset_limit:
        image_paths, labels = image_paths[:subset_limit], labels[:subset_limit]
    logging.info(f"Loaded {len(image_paths)} images.")
    return image_paths, labels

def preprocess_images(image_paths, labels, image_size=(224, 224)):
    """Preprocess images and create TensorFlow Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, image_size) / 255.0  # Normalize to [0, 1]
        return img, label

    dataset = dataset.map(preprocess)
    return dataset

def create_model(num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(base_dir, model_path, subset_limit=1000):
    image_paths, labels = load_images_with_labels(base_dir, subset_limit)

    # Map labels to integers
    class_names = sorted(set(labels))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([label_map[label] for label in labels])

    # Split into train and test
    dataset_size = len(image_paths)
    split_index = int(dataset_size * 0.8)
    train_paths, test_paths = image_paths[:split_index], image_paths[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Create datasets
    train_dataset = preprocess_images(train_paths, train_labels)
    test_dataset = preprocess_images(test_paths, test_labels)

    # Add data augmentation
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    train_dataset = train_dataset.map(lambda x, y: (augmentation(x), y))
    train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Create and train model
    num_classes = len(class_names)
    model = create_model(num_classes)

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    logging.info("Starting model training...")
    model.fit(train_dataset, validation_data=test_dataset, epochs=20, callbacks=callbacks)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
