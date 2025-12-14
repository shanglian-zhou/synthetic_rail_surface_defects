import os
import cv2
import numpy as np
import json
import shutil
from glob import glob
from tqdm import tqdm

# === Shared Definitions ===
label_folder = 'labels'
image_folder = 'images'
DEFECT_COLOR_BGR = (0, 0, 255)  # red
RESIZE_TO = (256, 256)

# === Combined Output Folder ===
combined_output_root = '/Volumes/Sandisk/data/RSDDs/controlnet_combined'
source_dir = os.path.join(combined_output_root, 'source')
target_dir = os.path.join(combined_output_root, 'target')
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

prompt_list = []

# === Define Dataset Groups ===
datasets = [
    {
        "root_dir": "/Volumes/Sandisk/data/RSDDs",
        "categories": ['type_1_cropped', 'type_2_cropped'],
        "prefix_map": {
            "type_1_cropped": "type_1_",
            "type_2_cropped": "type_2_"
        }
    },
    {
        "root_dir": "/Volumes/Sandisk/data/RSDDs/Synthetic",
        "categories": ['merged'],
        "prefix_map": {
            "merged": "syn_"
        }
    }
]

# === Processing Loop ===
for dataset in datasets:
    root_dir = dataset["root_dir"]
    categories = dataset["categories"]
    prefix_map = dataset["prefix_map"]

    for category in categories:
        print(f"\n🔧 Processing category: {category}")

        prefix = prefix_map.get(category, "")
        label_dir = os.path.join(root_dir, category, label_folder)
        image_dir = os.path.join(root_dir, category, image_folder)

        label_paths = sorted(glob(os.path.join(label_dir, '*.png')))

        for label_path in tqdm(label_paths):
            file_name = os.path.basename(label_path)
            new_file_name = f"{prefix}{file_name}"

            # === Step 1: Load and resize label ===
            label_img = cv2.imread(label_path)
            label_resized = cv2.resize(label_img, RESIZE_TO, interpolation=cv2.INTER_AREA)

            # === Step 2: Generate Canny edge from resized label ===
            defect_mask = cv2.inRange(label_resized, np.array(DEFECT_COLOR_BGR), np.array(DEFECT_COLOR_BGR))
            canny = cv2.Canny(defect_mask, 50, 150)
            canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

            # === Step 3: Save canny RGB as source ===
            final_source_path = os.path.join(source_dir, new_file_name)
            cv2.imwrite(final_source_path, canny_rgb)

            # === Step 4: Load and resize target image ===
            image_path = os.path.join(image_dir, file_name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb_resized = cv2.resize(img_rgb, RESIZE_TO, interpolation=cv2.INTER_AREA)
                final_target_path = os.path.join(target_dir, new_file_name)
                cv2.imwrite(final_target_path, cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR))
            else:
                print(f"⚠️ Image not found: {image_path}")
                continue

            # === Step 5: Add to prompt list ===
            prompt_list.append({
                "source": f"source/{new_file_name}",
                "target": f"target/{new_file_name}",
                "prompt": "rail track surface defect with spalling"
            })

# === Step 6: Save prompt.json ===
prompt_path = os.path.join(combined_output_root, "prompt.json")
with open(prompt_path, "w") as f:
    json.dump(prompt_list, f, indent=2)

print(f"\n✅ Done! Total images processed: {len(prompt_list)}")
print(f"📁 Output saved to: {combined_output_root}")
