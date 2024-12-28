import tensorflow as tf
import os
import logging
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import TRAIN_DIR, VALIDATION_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_model(num_classes):
    """Create a Convolutional Neural Network for multiclass classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Adding dropout for regularization
        layers.Dense(200, activation='softmax')  # 200 classes for output
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate
        loss='categorical_crossentropy',  # Loss function for multi-class classification
        metrics=['accuracy']  # Accuracy metric
    )
    return model

def train_and_save_model(base_dir, model_path, subset_limit=None):
    """Train the model and save it to the specified path."""
    logging.info("Setting up ImageDataGenerator with augmentation...")
    # Setup ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,               # Normalize images to [0, 1]
        rotation_range=30,               # Randomly rotate images
        width_shift_range=0.2,           # Randomly shift images horizontally
        height_shift_range=0.2,          # Randomly shift images vertically
        shear_range=0.2,                 # Randomly shear images
        zoom_range=0.2,                  # Random zoom
        horizontal_flip=True,            # Random horizontal flip
        fill_mode='nearest'              # Fill missing pixels after transformation
    )
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Load the dataset using the directory structure where subdirectories are the class labels
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # Directory with training images
        target_size=(224, 224), # Resize images to 224x224
        batch_size=32,         # Use a batch size that fits your memory
        class_mode='categorical' # 'categorical' for multi-class classification
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, # Directory with validation images
        target_size=(224, 224),    # Resize images to 224x224
        batch_size=32,
        class_mode='categorical'   # 'categorical' for multi-class classification
    )

    # Get number of classes and class names
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}  # Reverse the dictionary to get class names
    num_classes = len(class_names)
    logging.info(f"Found {num_classes} classes: {class_names[:5]}")

    # Apply subset limit if provided
    if subset_limit:
        steps_per_epoch = subset_limit // 32
    else:
        steps_per_epoch = len(train_generator)

    # Create the model
    model = create_model(num_classes)

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
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    # Save the model after training
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    return class_names  # Return class names for prediction
