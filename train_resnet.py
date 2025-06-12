import os
import shutil # Important for saving misclassified images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# === Set constants ===
IMG_SIZE = 224
BATCH_SIZE = 32
# We found that longer training with the right callbacks is key
EPOCHS = 5 

# --- IMPORTANT: Point to your leak-free dataset folder ---
# This should be the folder created by the 'infer_and_split_groups.py' script
train_dir = "dataset_split_inferred/train"
val_dir = "dataset_split_inferred/val"
test_dir = "dataset_split_inferred/test"

# === Image data generators ===
# Using enhanced data augmentation to build a more robust model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

# Validation and test generators should not be augmented, only rescaled
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# === Compute class weights ===
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# === Load ResNet50 base model ===
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# === Add custom layers with Dropout for regularization ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)  # Using the optimal dropout rate we found
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)  # Using the optimal dropout rate we found
x = Dense(64, activation="relu")(x)
x = Dense(train_generator.num_classes, activation="softmax")(x)

# === Create model ===
model = Model(inputs=base_model.input, outputs=x)

# === Callbacks optimized for confidence and stability ===
# Monitoring 'val_loss' to ensure the model trains until it is truly confident
lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5,
    verbose=1,
    mode='min'
)

stop_callback = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, # Increased patience to allow the model to stabilize
    verbose=1,
    mode='min',
    restore_best_weights=True # Restores weights from the epoch with the best val_loss
)

# === Compile model ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# === Train model ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[lr_callback, stop_callback]
)

# === Plot and Save training history ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history["precision"], label="Train")
plt.plot(history.history["val_precision"], label="Val")
plt.title("Precision")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history["recall"], label="Train")
plt.plot(history.history["val_recall"], label="Val")
plt.title("Recall")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# === Evaluate on test set ===
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
test_metrics_text = f"""
Test Accuracy: {test_acc:.4f}
Test Precision: {test_precision:.4f}
Test Recall: {test_recall:.4f}
"""
print(test_metrics_text)

# === Classification Report ===
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

report_text = classification_report(y_true, y_pred_classes, target_names=class_names)
print("\nClassification Report:")
print(report_text)

with open("model_performance.txt", "w") as f:
    f.write("--- Test Set Metrics ---\n")
    f.write(test_metrics_text)
    f.write("\n\n--- Classification Report ---\n")
    f.write(report_text)
print("✅ Performance report saved to model_performance.txt")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# === Save model ===
model.save("my_resnet_model.keras")
print("✅ Model saved as my_resnet_model.keras")

# === Generalized Error Analysis to find all misclassifications ===
print("\n--- Analyzing Misclassifications ---")
misclassified_dir = "misclassified_images"
if os.path.exists(misclassified_dir):
    shutil.rmtree(misclassified_dir)
os.makedirs(misclassified_dir)

test_generator.reset()

predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
filenames = test_generator.filenames

error_count = 0
for i in range(len(filenames)):
    if predicted_classes[i] != true_classes[i]:
        error_count += 1
        true_label = class_names[true_classes[i]]
        predicted_label = class_names[predicted_classes[i]]
        
        new_filename = f"TRUE_{true_label}_PRED_{predicted_label}_{os.path.basename(filenames[i])}"
        original_path = os.path.join(test_dir, filenames[i])
        destination_path = os.path.join(misclassified_dir, new_filename)
        
        shutil.copy(original_path, destination_path)

if error_count == 0:
    print("✅ No misclassifications found on the test set!")
else:
    print(f"✅ Saved {error_count} misclassified images to the '{misclassified_dir}' folder.")