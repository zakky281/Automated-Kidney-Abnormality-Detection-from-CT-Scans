import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import zipfile
import io

# --- CONFIGURATION ---
st.set_page_config(layout="wide")

MODEL_PATH = 'my_model.keras'
IMAGE_SIZE = (224, 224)
TRAIN_DIR = "split_dataset"

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_keras_model():
    """Loads the Keras model from the specified path."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.warning(f"Please ensure the model file '{MODEL_PATH}' is in the correct directory.")
        return None

@st.cache_data
def get_class_names(train_dir):
    """Retrieves class names from the training directory structure."""
    if not os.path.exists(train_dir):
        st.error(f"Training directory not found at '{train_dir}'. Cannot determine class names.")
        return None
    try:
        temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        temp_generator = temp_datagen.flow_from_directory(
            train_dir, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', shuffle=False
        )
        class_indices = temp_generator.class_indices
        class_names_map = {v: k for k, v in class_indices.items()}
        class_names = [class_names_map[i] for i in sorted(class_names_map.keys())]
        
        # Debugging output
        print("Class Names:", class_names)
        print("Number of classes:", len(class_names))
        
        return class_names
    except Exception as e:
        st.error(f"Failed to read class names from directory: {e}")
        return None

def predict(model, pil_image, class_names):
    """Processes the image and returns the predicted class and confidence."""
    if not class_names:
        raise ValueError("The class names list is empty or None. Check the dataset and ensure it's loaded correctly.")
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img = pil_image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_rescaled = img_array / 255.0
    img_array_expanded = tf.expand_dims(img_array_rescaled, 0)

    predictions = model.predict(img_array_expanded)
    
    # Debugging output
    print("Predictions shape:", predictions.shape)
    print("Predictions:", predictions)

    predicted_index = np.argmax(predictions[0])

    # Check if the predicted index is out of bounds
    if predicted_index >= len(class_names):
        raise ValueError(f"Predicted index {predicted_index} is out of bounds for class names list of length {len(class_names)}.")
    
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence

# --- INITIALIZATION ---
model = load_keras_model()
CLASS_NAMES = get_class_names(TRAIN_DIR)

# --- GUI LAYOUT ---
st.title("🔬 Kidney Disease Classification")
st.markdown("Upload kidney CT scan images individually or as a `.zip` file to classify them using the trained ResNet50 model.")

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.header("About")
    st.info("This application uses a custom-trained ResNet50 model to classify kidney CT scan images.")
    st.header("How to Use")
    st.markdown("""
    1.  **Upload:** Drag and drop images or a ZIP file into the uploader.
    2.  **Predict:** The model classifies each image automatically.
    3.  **Clear:** Click the 'Clear All' button to remove all uploads and start over.
    """)

# --- MAIN APP LOGIC ---
if model is None or CLASS_NAMES is None:
    st.warning("Application cannot run. Please check the error messages above.")
else:
    # --- STATE INITIALIZATION FOR THE RESET TRICK ---
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # --- CALLBACK FUNCTION TO INCREMENT THE KEY ---
    def increment_uploader_key():
        st.session_state.uploader_key += 1

    # --- FILE UPLOADER & CLEAR BUTTON ---
    uploaded_files = st.file_uploader(
        "Choose image(s) or a ZIP file...",
        type=["jpg", "jpeg", "png", "bmp", "zip"],
        accept_multiple_files=True,
        # The key is now dynamic, based on the counter in session_state
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    st.button(
        "Clear All Uploads",
        # The callback now just increments the counter, forcing a new key on rerun
        on_click=increment_uploader_key,
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files
    )

    if uploaded_files:
        st.header("Prediction Results")
        # --- PROCESSING LOGIC FOR IMAGES AND ZIPS ---
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.zip'):
                st.markdown(f"--- \n### Images from: `{uploaded_file.name}`")
                images_in_zip = 0
                with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
                    for filename in z.namelist():
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not filename.startswith('__MACOSX/'):
                            images_in_zip += 1
                            try:
                                image_bytes = z.read(filename)
                                image = Image.open(io.BytesIO(image_bytes))
                                

                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(image, caption=f"Inside ZIP: {filename}", use_container_width=True)
                                with col2:
                                    with st.spinner(f'Classifying {filename}...'):
                                        label, confidence = predict(model, image, CLASS_NAMES)
                                        st.success(f"**Predicted Class:** {label}")
                                        st.info(f"**Confidence:** {confidence:.2f}%")
                                        st.progress(int(confidence))
                            except UnidentifiedImageError:
                                st.warning(f"Skipped '{filename}' as it could not be identified as a valid image.")
                                continue
                if images_in_zip == 0:
                    st.warning(f"No valid images (.png, .jpg, etc.) found in `{uploaded_file.name}`.")
            else:
                try:
                    st.markdown("---")
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    with col2:
                        with st.spinner('Classifying...'):
                            label, confidence = predict(model, image, CLASS_NAMES)
                            st.success(f"**Predicted Class:** {label}")
                            st.info(f"**Confidence:** {confidence:.2f}%")
                            st.progress(int(confidence))
                except UnidentifiedImageError:
                    st.warning(f"Skipped '{uploaded_file.name}' as it is not a valid image file.")
