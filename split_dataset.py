import splitfolders

# Absolute path to your dataset
input_folder = r"D:\SEM 6\Sinyal\ResNet\data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

# This folder will be created in your current directory
output_folder = "dataset_split"

# 70% training, 20% validation, 10% testing
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1))

print("✅ Dataset split completed!")