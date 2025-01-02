import os
import random
from typing import List, Dict
import shutil

from constants import BASE_DIR, SUBSET_LIMIT, TRAIN_DIR, VALIDATION_DIR

# Constants to store in constants.py
HAND_POSITION = ["L"]
CAPTURE_TYPE = ["IN", "BP"]
RESOLUTION = ["1200PPI", "1106PPI"]

NEGATIVE_PRIORITY = 0
# Priority values for filtering
PRIORITY = {
    "hand_position": {"L": 10, "R": 1},
    "capture_type": {"IN": 2, "BP": 1},
    "resolution": {"1200PPI": 2, "1106PPI": 1},
}

# Function to parse filenames and filter based on whitelist criteria
def parse_and_filter_filenames(
    master_directory: str, whitelist: Dict[str, List[str]], priority: Dict[str, Dict[str, int]]
) -> List[str]:
    """
    Parse filenames from a master directory structure and filter based on whitelist criteria.

    :param master_directory: Master directory containing subject folders with fingerprint image files.
    :param whitelist: Dictionary with sections of filenames to filter.
    :param priority: Dictionary with priority values for filtering.
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
            
            if len(filtered_files) >= SUBSET_LIMIT:
                return filtered_files

            parts = filename.split("_")

            # Map parts of the filename to whitelist keys
            file_info = {
                "hand_position": parts[2],
                "capture_type": parts[4],
                "resolution": parts[5],
            }

            # Calculate priority sum for the file
            priority_sum = 0
            for key in file_info:
                if key in priority and file_info[key] in priority[key]:
                    priority_sum += priority[key][file_info[key]]
                else:
                    priority_sum += NEGATIVE_PRIORITY

            # max_priority = max(
            #     max(priority[key].values()) for key in file_info if key in priority
            # )

            # # Check if criteria are satisfied and priority sum is sufficient
            # if (
            #     any(file_info[key] in values for key, values in whitelist.items() if key in file_info)
            #     and priority_sum >= max_priority
            # ):
            filtered_files.append(os.path.join(subject_path, filename))
            
    return filtered_files

# Function to copy files to specified directories
def copy_files(file_list: List[str], destination_folder: str):
    """
    Copy files to a specified destination folder.

    :param file_list: List of file paths to copy.
    :param destination_folder: Destination folder to copy files into.
    """
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)  # Remove the directory and its contents
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_list:
        file_dir = os.path.basename(file_path).split("_")[0]
        path = "/".join([destination_folder,file_dir])

        os.makedirs(path, exist_ok=True)

    for file_path in file_list:
        file_dir = os.path.basename(file_path).split("_")[0]
        path = "/".join([destination_folder,file_dir,os.path.basename(file_path)])
        shutil.copy(file_path, path)

# Example usage
def main():
    master_directory = BASE_DIR  # Replace with your dataset path
    train_output_directory = TRAIN_DIR  # Replace with train output directory

    # Load filters from constants (simulated here, replace with actual imports)
    whitelist_filters = {
        "hand_position": HAND_POSITION,
        "capture_type": CAPTURE_TYPE,
        "resolution": RESOLUTION,
    }

    # Parse and filter filenames
    filtered_files = parse_and_filter_filenames(master_directory, whitelist_filters, PRIORITY)

    # Print the results
    print(f"Train files: {len(filtered_files)}")

    # Copy files to respective directories
    copy_files(filtered_files, train_output_directory)
    print("Creation done")

if __name__ == "__main__":
    main()
