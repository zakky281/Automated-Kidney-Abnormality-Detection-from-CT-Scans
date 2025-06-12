import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Make sure this path points to your dataset, just like in the training script
DATASET_PATH = os.path.join("data", "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

def create_dataframe(dataset_path):
    """Creates a pandas DataFrame of image paths and labels."""
    all_images = []
    all_labels = []

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                all_images.append(img_path)
                all_labels.append(class_name)
    
    return pd.DataFrame({"image": all_images, "label": all_labels})

def show_sample_images(df):
    """Displays one random sample image from each class."""
    print("[INFO] Displaying sample images from each class...")
    plt.figure(figsize=(15, 5))
    
    # Find the unique classes and get one sample from each
    unique_labels = df['label'].unique()
    for i, label in enumerate(unique_labels):
        sample_df = df[df['label'] == label].sample(1)
        image_path = sample_df['image'].iloc[0]
        image = Image.open(image_path)
        
        plt.subplot(1, len(unique_labels), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Class: {label}")
        plt.axis('off')
        
    plt.suptitle("Sample Images from Each Class", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_class_distribution(df, title):
    """Plots the class distribution of the entire dataset using a bar chart."""
    print(f"[INFO] Plotting: {title}")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df, order=df['label'].value_counts().index, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=0)
    plt.show()


if __name__ == "__main__":
    # 1. Create the DataFrame from your dataset folder
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at '{DATASET_PATH}'. Please check the path.")
    else:
        main_df = create_dataframe(DATASET_PATH)
        print("Dataset Information:")
        print(main_df.info())
        print("\nValue Counts:")
        print(main_df['label'].value_counts())

        # 2. Show a random sample image from each class
        show_sample_images(main_df)

        # 3. Plot the overall class distribution
        plot_class_distribution(main_df, "Overall Class Distribution in the Dataset")