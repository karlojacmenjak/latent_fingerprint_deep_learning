import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    """Create a Convolutional Neural Network for multiclass classification."""
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Regularization
        layers.Dense(num_classes, activation='softmax')  # Multiclass classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(base_dir, model_path, subset_limit=None):
    """Train the model and save it to the specified path."""
    logging.info("Loading and preprocessing images...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        image_size=(224, 224),
        batch_size=32,
        labels="inferred",
        label_mode="int",  # Integer labels
        shuffle=True
    )
    
    # Extract class names
    class_names = dataset.class_names
    num_classes = len(class_names)
    logging.info(f"Found {num_classes} classes: {class_names}")

    # Apply subset limit if provided
    if subset_limit:
        dataset = dataset.take(subset_limit // 32)

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    logging.info("Creating model...")
    model = create_model(num_classes)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    logging.info("Starting model training...")
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return class_names
