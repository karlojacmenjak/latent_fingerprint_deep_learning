import os
import random
import shutil
from sklearn.model_selection import train_test_split

from constants import BASE_DIR, SUBSET_LIMIT, VALIDATION_DIR, TRAIN_DIR

# Paths to the original dataset and the new directories
original_dataset_dir = BASE_DIR
train_dir = TRAIN_DIR
validation_dir = VALIDATION_DIR

# List all class subdirectories
class_dirs = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]

# Split data by class
for class_dir in class_dirs:
    class_path = os.path.join(original_dataset_dir, class_dir)
    # List all files in the class folder
    all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    # Split files into train and validation

    all_files = all_files[:SUBSET_LIMIT]
    train_files, validation_files = train_test_split(all_files, shuffle=True, test_size=0.2, random_state=42)
    
    # Create subdirectories for class in train and validation directories
    os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_dir), exist_ok=True)
    
    # Move files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_dir, file))

    for file in validation_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(validation_dir, class_dir, file))

print("Data splitting completed.")