import tensorflow as tf
import numpy as np
import os
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# os.environ["OMP_NUM_THREADS"] = "2"  # Adjust based on your CPU core count
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(2)

# Ensure that TensorFlow only uses as much GPU memory as needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for GPUs to dynamically allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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

def load_images(base_dir, subset_limit=None):
    image_paths = []
    labels = []
    
    print("Loading images from directory:", base_dir)
    for label_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, label_folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {label_folder}")
            for img_name in os.listdir(folder_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(folder_path, img_name)
                    image_paths.append(img_path)
                    labels.append(label_folder)  # Use folder name as label
    
    # Limit the dataset if subset_limit is specified
    if subset_limit:
        image_paths = image_paths[:subset_limit]
        labels = labels[:subset_limit]
    
    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels

def preprocess_images(image_paths, batch_size=32):
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
        
        # Batch processing to avoid memory overload
        if len(images) >= batch_size:
            yield np.array(images)
            images = []
    
    if images:
        yield np.array(images)  # Yield any remaining images

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

# Limit the dataset size for faster experiments
SUBSET_LIMIT = 1000  # Use only 1000 images

# Load and preprocess images
image_paths, labels = load_images(BASE_DIR, subset_limit=SUBSET_LIMIT)

# Encode labels to binary format (for binary classification)
le = LabelEncoder()
labels = le.fit_transform(labels)

# Preprocess the images and store the results in lists
images_list = []
for image_batch in preprocess_images(image_paths):
    images_list.extend(image_batch)

# Ensure images_list and labels are the same length
assert len(images_list) == len(labels), f"Mismatch between number of images ({len(images_list)}) and labels ({len(labels)})"

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_list, labels, test_size=0.2, random_state=42)

# Normalize images to [0, 1] range
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# Create the model
model = create_model()

# Define progress bar during training
class ProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} finished: Accuracy: {logs['accuracy']:.4f}, Loss: {logs['loss']:.4f}")

# Train the model with the use of a progress bar and handle large batch processing
progress_bar = ProgressBar()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Training the model...")
model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=16, 
    verbose=2, 
    callbacks=[progress_bar, early_stopping]
)

# Save the model
MODEL_SAVE_PATH = "fingerprint_model.h5"
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
