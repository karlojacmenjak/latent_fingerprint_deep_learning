import tensorflow as tf
import numpy as np
import os
import cv2
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tqdm import tqdm  # Import tqdm for progress bars

# Define image size
IMG_SIZE = (224, 224)  # Resize images to 224x224

# Define the base directory for images
BASE_DIR = './fingerprints/images/latent/png'  # Modify this to your actual directory

# Image augmentation
aug = A.Compose([
    A.RandomCrop(width=224, height=224),  # Crop to the desired size
    A.HorizontalFlip(),  # Random horizontal flip
    A.RandomBrightnessContrast(p=0.2),  # Random brightness/contrast adjustment
    A.HueSaturationValue(p=0.3),  # Random hue/saturation adjustment
])

def load_images(base_dir):
    image_paths = []
    labels = []
    
    print("Loading images from directory:", base_dir)
    # Traverse through the folders (each folder corresponds to an image)
    for label_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, label_folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {label_folder}")
            for img_name in os.listdir(folder_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(folder_path, img_name)
                    image_paths.append(img_path)
                    labels.append(label_folder)  # Use folder name as label
    
    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels

def preprocess_images(image_paths):
    images = []
    print("Preprocessing images...")
    # Apply augmentation and resize images
    for img_path in tqdm(image_paths, desc="Preprocessing images", ncols=100):
        img = cv2.imread(img_path)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, IMG_SIZE)  # Resize to match input size
        
        # Apply augmentation
        augmented = aug(image=img)
        img = augmented['image']
        
        images.append(img)
    
    return np.array(images)

def create_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess images
image_paths, labels = load_images(BASE_DIR)
images = preprocess_images(image_paths)
labels = np.array(labels)

# Convert labels to binary (if two classes)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize images to [0, 1] range
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create and train the model
model = create_model()

# Use TensorFlow's built-in progress bar for training
class ProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} finished: Accuracy: {logs['accuracy']:.4f}, Loss: {logs['loss']:.4f}")

# Train the model and display a progress bar for each epoch
print("Training the model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0, callbacks=[ProgressBar()])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
