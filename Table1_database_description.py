# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:53:46 2025

@author: nicolas
"""

import os
import numpy as np

# Directorio con las etiquetas
label_dir = "Placenta P007 - P053 labels/"
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

# Orden de clases
class_names = {
    0: "Unlabeled",               # black
    1: "Artery",                  # red
    2: "Specular reflection",     # yellow
    3: "Stroma",                  # green
    4: "Vein",                    # blue
    5: "Suture",                  # purple
    6: "Umbilical cord"           # orange
}

num_classes = len(class_names)
class_image_count = {i: 0 for i in range(num_classes)}
class_pixel_count = {i: 0 for i in range(num_classes)}

for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)
    label = np.load(label_path)

    unique_classes = np.unique(label)
    for cls in unique_classes:
        mask = label == cls
        pixel_count = np.sum(mask)
        class_pixel_count[cls] += pixel_count
        class_image_count[cls] += 1

print(f"{'Class':<25} {'#Images':>8} {'Total Pixels':>15} {'Avg. Pixels/Image':>20}")
print("-" * 70)

for cls in range(num_classes):
    name = class_names[cls]
    img_count = class_image_count[cls]
    pix_count = class_pixel_count[cls]
    avg = pix_count / img_count if img_count > 0 else 0
    print(f"{name:<25} {img_count:>8} {pix_count:>15,} {avg:>20,.2f}")
