import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
# ✅ CHOOSE WHICH MODEL TO EVALUATE:
# MODEL_PATH = "baseline_model.keras"
MODEL_PATH = "baseline_model.keras" # Currently set to evaluate the advanced model

# --- Path to your test dataset ---
test_dir = "dataset_split_inferred/test"

# --- Set constants ---
IMG_SIZE = 224
BATCH_SIZE = 32 # Should match the batch size used in training for consistency

# --- Load the saved model ---
print(f"[INFO] Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found at '{MODEL_PATH}'.")
    print("Please run the appropriate training script first to create the model file.")
    exit()
    
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- Prepare the test data generator ---
# It's crucial that preprocessing here matches the training preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT: Do not shuffle test data for evaluation
)

print(f"\n--- Evaluating Model: {os.path.basename(MODEL_PATH)} ---")

# === Get Predictions from the Model ===
# Reset the generator to be sure it's at the start
test_generator.reset()
predictions = model.predict(
    test_generator,
    steps=len(test_generator), # Ensure all samples are predicted
    verbose=1
)

# Get the predicted class indices
y_pred_classes = np.argmax(predictions, axis=1)
# Get the true class indices
y_true = test_generator.classes
# Get the class names in the correct order
class_names = list(test_generator.class_indices.keys())


# === Generate and Print Classification Report ===
print("\n📋 Classification Report:")
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)


# === Generate and Plot Confusion Matrix ===
print("\n📊 Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix for {os.path.basename(MODEL_PATH)}", fontsize=16)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)

# Save the plot with a descriptive name
output_filename = f"evaluation_{os.path.splitext(os.path.basename(MODEL_PATH))[0]}.png"
plt.savefig(output_filename)
print(f"✅ Confusion matrix plot saved as '{output_filename}'")
plt.show()
