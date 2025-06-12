import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
import imagehash

# --- Configuration ---
# Path to your original, unsplit dataset
input_folder = r"D:\SEM 6\Sinyal\ResNet\data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
# Path for the new, correctly split dataset
output_folder = "dataset_split_inferred"

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# --- IMPORTANT: TUNABLE PARAMETER ---
# The hash difference threshold to decide if an image is from a new patient.
# A small difference (0-5) means images are very similar.
# A large difference (> 10) means they are likely different.
# You may need to experiment with this value. A good start is 5 or 10.
HASH_DIFFERENCE_THRESHOLD = 5

print("Starting inferred group dataset split...")

# This will hold our final groups of files, e.g., [['patient1_file1', 'patient1_file2'], ['patient2_file1', ...]]
all_patient_groups = []

# 1. Iterate through each class folder (Cyst, Normal, etc.)
for class_name in sorted(os.listdir(input_folder)):
    class_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    
    print(f"\nProcessing class: {class_name}")

    # Get a list of image files and sort them to ensure correct sequence
    files = sorted(os.listdir(class_path))
    if not files:
        continue
        
    class_groups = []
    current_group = []
    
    # Use the first file to start the first group
    last_hash = imagehash.phash(Image.open(os.path.join(class_path, files[0])))
    current_group.append(os.path.join(class_path, files[0]))

    # 2. Iterate through the rest of the files to find boundaries
    for i in range(1, len(files)):
        filepath = os.path.join(class_path, files[i])
        current_hash = imagehash.phash(Image.open(filepath))
        
        # Compare the hash of the current image to the previous one
        hash_diff = current_hash - last_hash
        
        if hash_diff > HASH_DIFFERENCE_THRESHOLD:
            # BIG JUMP! A new patient is starting.
            # Finish the old group and start a new one.
            class_groups.append(current_group)
            current_group = []
            print(f"  - Detected new patient group. Previous group size: {len(class_groups[-1])} images. Hash diff: {hash_diff}")
        
        current_group.append(filepath)
        last_hash = current_hash
        
    # Add the last group
    if current_group:
        class_groups.append(current_group)
    
    print(f"  - Finished final group. Size: {len(current_group)} images.")
    print(f"  - Inferred {len(class_groups)} patient groups for class '{class_name}'.")
    all_patient_groups.extend(class_groups)

print(f"\nTotal inferred patient groups across all classes: {len(all_patient_groups)}")

# 3. Split the list of unique patient groups
train_groups, val_test_groups = train_test_split(
    all_patient_groups,
    test_size=(val_ratio + test_ratio),
    random_state=42
)
val_groups, test_groups = train_test_split(
    val_test_groups,
    test_size=(test_ratio / (val_ratio + test_ratio)),
    random_state=42
)

print(f"Splitting into: {len(train_groups)} train, {len(val_groups)} validation, {len(test_groups)} test groups.")

# 4. Create output directories and copy files
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

split_mapping = {
    'train': train_groups,
    'val': val_groups,
    'test': test_groups
}

for split_name, group_list in split_mapping.items():
    for group in group_list:
        if not group:
            continue
        
        # Get class name from the first file in the group
        class_name = os.path.basename(os.path.dirname(group[0]))
        
        dest_dir = os.path.join(output_folder, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        for file_path in group:
            shutil.copy(file_path, dest_dir)
            
print("\n✅ Inferred group dataset split completed successfully!")
print(f"New leak-free dataset is in the '{output_folder}' directory.")