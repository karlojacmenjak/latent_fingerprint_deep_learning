import os
import logging


from keras import layers, models
# fix for imports: https://stackoverflow.com/a/78504065
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.saving import save_model


from constants import TRAIN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_model():
    """Create a Convolutional Neural Network for multiclass classification."""
    model = models.Sequential([
        layers.Input((224, 224, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Adding dropout for regularization
        layers.Dense(41, activation='softmax')  # 200 classes for output
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Adjust learning rate
        loss='categorical_crossentropy',  # Loss function for multi-class classification
        metrics=['accuracy']  # Accuracy metric
    )
    return model

def train_and_save_model(base_dir, model_path, subset_limit=None):
    """Train the model and save it to the specified path."""
    logging.info("Setting up ImageDataGenerator with augmentation...")

    seed = int.from_bytes(os.urandom(4), 'little')
    train_dataset, validation_dataset = image_dataset_from_directory(
        directory = TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(224, 224),
        seed=seed,
        validation_split=0.2,
        subset="both",
        color_mode="grayscale",
        verbose=True,
    )
    
    label_map = (train_dataset.class_names)

    model = create_model()

    # Define EarlyStopping callback
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=5,          # Wait 5 epochs for improvement before stopping
            restore_best_weights=True
        )
    ]

    # Train the model using the data generator
    logging.info("Starting model training...")
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=validation_dataset,
        callbacks=callbacks
    )

    # Save the model after training
    save_model(model,model_path, overwrite=True)
    logging.info(f"Model saved to {model_path}")

    return label_map
