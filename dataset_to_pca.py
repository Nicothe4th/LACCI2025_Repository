# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:57:37 2025

@author: nicolas
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from spectral_tiffs import read_stiff
import re
import os

class HyperspectralDataset(Dataset):
    def __init__(self, image_dir, label_dir, augmentation=False, augmentation_ratio=0.1, padding_value=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        pattern = re.compile(r"P\d{3}\.tif$")
        self.tif_files = sorted([f for f in os.listdir(image_dir) if pattern.match(f)])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        
        assert len(self.tif_files) == len(self.label_files), "Número de imágenes y etiquetas no coincide"

        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.padding_value = padding_value

        # Select random indices for augmentation
        num_augmented = int(len(self.tif_files) * self.augmentation_ratio)
        self.augmented_indices = set(random.sample(range(len(self.tif_files)), num_augmented))

        # PCA settings
        self.n_components = 26

    def __len__(self):
        return len(self.tif_files)

    def __getitem__(self, idx):
        # Load hyperspectral image and label
        image, _, _, _ = read_stiff(os.path.join(self.image_dir, self.tif_files[idx]))
        label = np.load(os.path.join(self.label_dir, self.label_files[idx]))  # (1024, 1024)

        # Apply PCA to reduce to self.n_components
        Ny, Nx, Nz = image.shape
        flat = image.reshape((Nx * Ny, Nz), order='F')  # Shape: (H*W, bands)

        # Standardize before PCA
        flat_std = StandardScaler().fit_transform(flat)

        # Apply PCA
        pca = PCA(n_components=self.n_components)
        flat_pca = pca.fit_transform(flat_std)

        # Reshape back to image format (C, H, W)
        image_reduced = flat_pca.reshape(Ny, Nx, self.n_components, order='F')
        image_tensor = torch.tensor(image_reduced, dtype=torch.float32).permute(2, 0, 1)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0)

        # Apply data augmentation
        if self.augmentation and idx in self.augmented_indices:
            image_tensor, label_tensor = self.apply_augmentation(image_tensor, label_tensor)

        return image_tensor, label_tensor.squeeze(0)

    def apply_augmentation(self, image, label):
        """Aplica rotaciones y flips de forma consistente en imagen y etiquetas, con padding reflectante."""
        angle = random.uniform(-30, 30)
        
        # Flips aleatorios
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Padding reflectante antes de la rotación
        padding = 20
        pad_transform = T.Pad(padding, padding_mode='reflect')
        image = pad_transform(image)
        label = pad_transform(label)

        # Rotación
        image = TF.rotate(image, angle, fill=self.padding_value)
        label = TF.rotate(label, angle, fill=self.padding_value)

        # Recorte al tamaño original
        image = TF.center_crop(image, (1024, 1024))
        label = TF.center_crop(label, (1024, 1024))

        return image, label

def get_dataset(augmentation=False, augmentation_ratio=0.1, padding_value=0):
    return HyperspectralDataset(
        image_dir="Placenta P007 - P053 red blue/",
        label_dir="Placenta P007 - P053 labels/",
        augmentation=augmentation,
        augmentation_ratio=augmentation_ratio,
        padding_value=padding_value
    )

def get_splits(dataset, k=5):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    fold_size = len(dataset) // k
    folds = [(indices[i*fold_size:(i+1)*fold_size], indices[:i*fold_size] + indices[(i+1)*fold_size:]) for i in range(k)]
    return folds
