# -*- coding: utf-8 -*-
"""
Created on Sat May  3 21:49:48 2025

@author: nicolas
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import get_dataset
from dataset_to_MSI import get_dataset as get_dataset_MSI
from dataset_to_pca import get_dataset as get_dataset_pca
from model import UNet
import matplotlib.patches as mpatches  # Import for Patch
# Define label mapping
LABELS = [
    "Artery", "Specular reflection", "Stroma", "Vein", "Suture", "Umbilical cord"
]
LABEL_MAP = {label: i + 1 for i, label in enumerate(LABELS)}  # Start labeling from 1

LABEL_COLORS = {
    "Unlabeled": "black",
    "Artery": "red",
    "Specular reflection": "yellow",
    "Stroma": "green",
    "Vein": "blue",
    "Suture": "purple",
    "Umbilical cord": "orange",
}

COLOR_LIST = ["black"] + [LABEL_COLORS[label] for label in LABELS]  # Ensure black is first for unlabeled pixels

dataset = get_dataset()
datasetPCA = get_dataset_pca()
datasetMSI = get_dataset_MSI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ab = UNet(num_channels=8, num_classes=7).to(device)
model_ab.load_state_dict(torch.load("unet_fold1.pth"))
model_ab.eval()

model_pca = UNet(num_channels=26, num_classes=7).to(device)
model_pca.load_state_dict(torch.load("pca_unet_fold1.pth"))
model_pca.eval()

model_msi = UNet(num_channels=37, num_classes=7).to(device)
model_msi.load_state_dict(torch.load("MSI_unet_fold1.pth"))
model_msi.eval()
cmap = plt.matplotlib.colors.ListedColormap(COLOR_LIST)

with torch.no_grad():
    image, labels = dataset[0]
    image = image.unsqueeze(0).to(device)
    output = model_ab(image)
    predicted_ab = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    image, labels = datasetPCA[0]
    image = image.unsqueeze(0).to(device)
    output = model_pca(image)
    predicted_pca = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    image, labels = datasetMSI[0]
    image = image.unsqueeze(0).to(device)
    output = model_msi(image)
    predicted_msi = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

fig, axs = plt.subplots(2, 2, figsize=(10, 15))

axs[0,0].imshow(labels, cmap=cmap, vmin=0, vmax=len(LABELS))
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,0].set_title("(a) ground-truth",fontsize=14)

axs[0,1].imshow(predicted_ab, cmap=cmap, vmin=0, vmax=len(LABELS))
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].set_title("b) spectral unmixing-based Classification",fontsize=14)
#axs[1].set_ylabel(model_path)

handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LABEL_COLORS[label], markersize=10)
    for label in ["Unlabeled"] + LABELS
]

fig.legend(
    handles, ["Unlabeled"] + LABELS,
    loc='upper left',
    bbox_to_anchor=(0.85, 0.90),  # Slightly inset from top-left corner
    fontsize=14,
    frameon=False
)



axs[1,0].imshow(predicted_pca, cmap=cmap, vmin=0, vmax=len(LABELS))
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,0].set_title("(c) PCA-based Classification",fontsize=14)
#axs[1].set_ylabel(model_path)

axs[1,1].imshow(predicted_msi, cmap=cmap, vmin=0, vmax=len(LABELS))
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
axs[1,1].set_title("(d) raw-MSI Classification",fontsize=14)
#axs[1].set_ylabel(model_path)

plt.show()