import tensorflow as tf
import numpy as np
import os
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Image augmentation
aug = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.3),
])

def preprocess_images(image_paths, batch_size=32):
    for img_path in tqdm(image_paths, desc="Preprocessing images", ncols=100):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        augmented = aug(image=img)
        img = augmented['image']
        yield img / 255.0

def load_images(base_dir, subset_limit=None):
    image_paths = []
    labels = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.endswith('.png'):
                    # Extract the identifier before "_" in the filename
                    identifier = img.split('_')[0]
                    image_paths.append(os.path.join(folder_path, img))
                    labels.append(identifier)
    if subset_limit:
        image_paths, labels = image_paths[:subset_limit], labels[:subset_limit]
    logging.info(f"Loaded {len(image_paths)} images.")
    return image_paths, labels

def create_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(base_dir, model_path, subset_limit=1000):
    image_paths, labels = load_images(base_dir, subset_limit)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    images = list(preprocess_images(image_paths))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = create_model()

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    logging.info("Starting model training...")
    model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), 
              epochs=10, batch_size=16, callbacks=callbacks, verbose=2)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
