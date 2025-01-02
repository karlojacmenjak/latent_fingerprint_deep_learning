import os
import random
from typing import List, Dict
import shutil

from constants import BASE_DIR, TRAIN_DIR, VALIDATION_DIR

# Constants to store in constants.py
HAND_POSITION = ["L"]
CAPTURE_TYPE = ["IN", "BP"]
RESOLUTION = ["1200PPI", "1106PPI"]

# Function to parse filenames and filter based on whitelist criteria
def parse_and_filter_filenames(master_directory: str, whitelist: Dict[str, List[str]]) -> List[str]:
    """
    Parse filenames from a master directory structure and filter based on whitelist criteria.

    :param master_directory: Master directory containing subject folders with fingerprint image files.
    :param whitelist: Dictionary with sections of filenames to filter.
    :return: List of filtered filenames.
    """
    filtered_files = []

    for subject_folder in os.listdir(master_directory):
        subject_path = os.path.join(master_directory, subject_folder)

        if not os.path.isdir(subject_path):
            continue

        for filename in os.listdir(subject_path):
            if not filename.endswith(".png"):
                continue

            parts = filename.split("_")

            # Map parts of the filename to whitelist keys
            file_info = {
                "hand_position": parts[2],
                "capture_type": parts[4],
                "resolution": parts[5],
            }

            # Check if all criteria are satisfied
            if any(file_info[key] in values for key, values in whitelist.items() if key in file_info):
                filtered_files.append(os.path.join(subject_path, filename))

    return filtered_files

# Function to split data into train and validation sets
def split_train_validation(file_list: List[str], validation_split: float = 0.2):
    """
    Split file list into train and validation sets ensuring no overlap.

    :param file_list: List of files to split.
    :param validation_split: Fraction of data to use for validation.
    :return: Tuple of (train_files, validation_files).
    """
    random.shuffle(file_list)
    split_idx = int(len(file_list) * (1 - validation_split))
    train_files = file_list[:split_idx]
    validation_files = file_list[split_idx:]

    # Ensure no overlap between train and validation sets
    assert not set(train_files) & set(validation_files), "Train and validation sets overlap!"

    return train_files, validation_files

# Function to copy files to specified directories
def copy_files(file_list: List[str], destination_folder: str):
    """
    Copy files to a specified destination folder.

    :param file_list: List of file paths to copy.
    :param destination_folder: Destination folder to copy files into.
    """
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_list:
        shutil.copy(file_path, destination_folder)

# Example usage
def main():
    master_directory = BASE_DIR  # Replace with your dataset path
    train_output_directory = TRAIN_DIR  # Replace with train output directory
    validation_output_directory = VALIDATION_DIR  # Replace with validation output directory

    # Load filters from constants (simulated here, replace with actual imports)
    whitelist_filters = {
        "hand_position": HAND_POSITION,
        "capture_type": CAPTURE_TYPE,
        "resolution": RESOLUTION,
    }

    # Parse and filter filenames
    filtered_files = parse_and_filter_filenames(master_directory, whitelist_filters)

    # Split into train and validation sets
    train_files, validation_files = split_train_validation(filtered_files)

    # Print the results
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(validation_files)}")

    # Copy files to respective directories
    copy_files(train_files, train_output_directory)
    copy_files(validation_files, validation_output_directory)

if __name__ == "__main__":
    main()
