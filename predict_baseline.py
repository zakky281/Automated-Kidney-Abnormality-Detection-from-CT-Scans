import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
import os

# --- CONFIGURATION ---
# ✅ MODIFIED: This now points to the baseline model
MODEL_PATH = 'baseline_resnet50_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- LOAD MODEL ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Loading baseline model from {MODEL_PATH}...")
except Exception as e:
    print(f"[ERROR] Could not load model. Error: {e}")
    exit()

# --- TKINTER SETUP (HIDDEN WINDOW) ---
root = Tk()
root.withdraw()

print("\nOpening file chooser... Please select an image to classify with the BASELINE model.")
while True:
    file_path = filedialog.askopenfilename(
        title="Select an image to predict",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        break

    try:
        # --- LOAD & PREPROCESS IMAGE ---
        img = image.load_img(file_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- PREDICT ---
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_index = np.argmax(score)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence_percent = 100 * np.max(score)

        # --- PRINT RESULT ---
        print("\n" + "="*30)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence_percent:.2f}%")
        print("="*30)

    except Exception as e:
        print(f"[ERROR] Could not process file {file_path}. Error: {e}")

    print("\nOpening file chooser... Please select an image to classify.")