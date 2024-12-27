import os
import logging
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import pyperclip
import subprocess

class FingerprintPredictionUI:
    def __init__(self, master, model_path, image_results, base_dir):
        self.master = master
        self.master.title("Fingerprint Prediction UI")
        self.master.geometry("800x600")
        
        self.model_path = model_path
        self.image_results = image_results  # List of results: {'image': img_path, 'identifier': ID, 'prediction': status, 'confidence': score}
        self.base_dir = base_dir
        self.current_index = 0
        
        # UI Elements
        self.stats_label = ttk.Label(self.master, text="", font=("Arial", 12))
        self.stats_label.pack(pady=10)

        self.image_label = ttk.Label(self.master)
        self.image_label.pack(pady=20)

        self.info_label = ttk.Label(self.master, text="", font=("Arial", 14))
        self.info_label.pack(pady=10)
        
        self.copy_button = ttk.Button(self.master, text="Copy File Name", command=self.copy_file_name)
        self.copy_button.pack(pady=10)

        self.open_folder_button = ttk.Button(self.master, text="Open Folder", command=self.open_folder)
        self.open_folder_button.pack(pady=10)

        self.prev_button = ttk.Button(self.master, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side="left", padx=20)

        self.next_button = ttk.Button(self.master, text="Next", command=self.show_next_image)
        self.next_button.pack(side="right", padx=20)

        self.show_image(self.current_index)

    def copy_file_name(self):
        """Copy the file name of the currently displayed image to the clipboard."""
        result = self.image_results[self.current_index]
        img_path = result['image']
        file_name = os.path.basename(img_path)  # Get the file name from the image path
        pyperclip.copy(file_name)  # Copy to clipboard
        logging.info(f"Copied file name: {file_name}")
        
        # Display a message on the UI that the file name was copied
        self.info_label.config(text=f"{self.info_label.cget('text')}\nFile Name copied to clipboard!")

    def open_folder(self):
        """Open the subfolder corresponding to the current image's identifier."""
        current_result = self.image_results[self.current_index]
        identifier = current_result['identifier']
        folder_path = os.path.join(self.base_dir, identifier)
        
        if os.path.exists(folder_path):
            if os.name == 'nt':  # For Windows
                subprocess.run(['explorer', folder_path])
            elif os.name == 'posix':  # For macOS or Linux
                subprocess.run(['xdg-open', folder_path])
            else:
                logging.warning(f"Unable to open the folder on this OS: {os.name}")
        else:
            logging.warning(f"The folder for identifier {identifier} does not exist in {self.base_dir}")

    def show_image(self, index):
        """Display the image, prediction result, and confidence."""
        result = self.image_results[index]
        
        # Load and display image
        img_path = result['image']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert the image to PIL format
        img = img.resize((224, 224))  # Resize to fit the window

        img_tk = ImageTk.PhotoImage(img)  # Convert to Tkinter-compatible format
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to the image

        # Update info label with prediction and confidence
        self.info_label.config(
            text=f"ID: {result['identifier']}\nPrediction: {result['prediction']}\nConfidence: {result['confidence']*100:.2f}%"
        )

    def show_next_image(self):
        """Navigate to the next image in the list."""
        if self.current_index < len(self.image_results) - 1:
            self.current_index += 1
            self.show_image(self.current_index)

    def show_previous_image(self):
        """Navigate to the previous image in the list."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.current_index)


def run_ui(model_path, image_results, base_dir):
    """Runs the UI with the given model and prediction results."""
    root = tk.Tk()
    ui = FingerprintPredictionUI(root, model_path, image_results, base_dir)
    root.mainloop()

