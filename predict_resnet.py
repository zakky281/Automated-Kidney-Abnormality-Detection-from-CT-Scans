import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import tkinter as tk
from tkinter import filedialog

# === Constants ===
IMG_SIZE = 224

# --- Function to load class names ---
def get_class_names(train_dir="dataset_split_inferred/train"):
    """
    Retrieves a list of class names from the training directory structure.
    This ensures the prediction output is human-readable.
    """
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' not found.")
        print("Please ensure the script is run from the same root directory as your training script,")
        print("or provide the correct path to your training data.")
        return None
    
    # Create a temporary generator to safely access class indices
    temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    temp_generator = temp_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # The class_indices attribute is a dict mapping class names to integer indices
    class_indices = temp_generator.class_indices
    # We invert it to map from index to class name for our final output
    class_names = {v: k for k, v in class_indices.items()}
    
    print("✅ Class names loaded successfully.")
    return class_names

# --- Function to open file explorer and select an image ---
def select_image_file():
    """
    Opens a file explorer window for the user to select an image.
    Returns the path to the selected image.
    """
    # Create a Tkinter root window
    root = tk.Tk()
    # Hide the main window, we only want the dialog
    root.withdraw()
    
    # Open the file dialog, allowing specific image file types
    file_path = filedialog.askopenfilename(
        title="Select an Image File (or press Cancel to exit)",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
    )
    
    return file_path

# --- Function to preprocess the image ---
def preprocess_image(img_path):
    """
    Loads and preprocesses an image for model prediction.
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array_rescaled = img_array / 255.0
    img_array_expanded = np.expand_dims(img_array_rescaled, axis=0)
    return img_array_expanded

# --- Main prediction logic ---
def main(model_path="my_resnet_model.keras"):
    # === STEP 1: Load model and class names ONCE ===
    print("Loading model and class names, please wait...")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load model. {e}")
        return

    class_names = get_class_names()
    if class_names is None:
        return # Exit if class names can't be loaded

    # === STEP 2: Start the prediction loop ===
    while True:
        print("\n" + "="*50)
        # Let the user select an image via the file explorer
        image_path = select_image_file()

        # If the user closes the dialog or presses cancel, the path will be empty.
        # This is our condition to exit the loop.
        if not image_path:
            print("\nNo image selected. Thank you for using the predictor!")
            break # Exit the while loop

        # --- If an image was selected, proceed with prediction ---
        try:
            # Preprocess the selected image
            processed_image = preprocess_image(image_path)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the predicted class index and confidence score
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence_score = np.max(prediction)
            
            # Get the predicted class name
            predicted_class_name = class_names.get(predicted_class_index, "Unknown Class")
            
            # Display the result
            print("\n--- Prediction Result ---")
            print(f"🖼️  Image File: {os.path.basename(image_path)}")
            print(f"🧠  Predicted Class: {predicted_class_name}")
            print(f"🎯  Confidence: {confidence_score:.2%}")
            print("-------------------------\n")

        except Exception as e:
            print(f"\nAn error occurred while processing {os.path.basename(image_path)}: {e}")
            print("Please try again with a different image.")


if __name__ == "__main__":
    main()