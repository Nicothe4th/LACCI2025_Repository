#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:27:29 2025

@author: jnmc
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from spectral_tiffs import read_stiff, read_mtiff
import re
import os

class HyperspectralDataset(Dataset):
    def __init__(self, image_dir, label_dir, augmentation=False, augmentation_ratio=0.1, padding_value=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        pattern = re.compile(r"P\d{3}\.tif$")
        self.tif_files = sorted([f for f in os.listdir(image_dir) if pattern.match(f) ])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        
        
        assert len(self.tif_files) == len(self.label_files), "Número de imágenes y etiquetas no coincide"

        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.padding_value = padding_value

        # Seleccionar aleatoriamente imágenes para aplicar augmentation
        num_augmented = int(len(self.tif_files) * self.augmentation_ratio)
        self.augmented_indices = set(random.sample(range(len(self.tif_files)), num_augmented))

    def __len__(self):
        return len(self.tif_files)

    def __getitem__(self, idx):
        # Cargar imagen y etiquetas

        image, _, _, _ = read_stiff(os.path.join(self.image_dir, self.tif_files[idx]))
        label = np.load(os.path.join(self.label_dir, self.label_files[idx]))  # (1024, 1024)

        # Convertir a tensores de PyTorch
        image = torch.tensor(image, dtype=torch.float32).permute(dims=[2, 0, 1])  
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)  

        # Aplicar data augmentation solo a los índices seleccionados
        if self.augmentation and idx in self.augmented_indices:
            image, label = self.apply_augmentation(image, label)

        return image, label.squeeze(0)  # Quitar la dimensión extra del label al final

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
        padding = 20  # Cantidad de padding en píxeles (ajústalo según necesidad)
        pad_transform = T.Pad(padding, padding_mode='reflect')

        image = pad_transform(image)
        label = pad_transform(label)

        # Rotación
        image = TF.rotate(image, angle, fill=self.padding_value)
        label = TF.rotate(label, angle, fill=self.padding_value)

        # Recortar para restaurar tamaño original
        image = TF.center_crop(image, (1024, 1024))
        label = TF.center_crop(label, (1024, 1024))

        return image, label


def get_dataset(augmentation=False, augmentation_ratio=0.1, padding_value=0):
    """Carga el dataset completo con un porcentaje específico de augmentation y padding reflectante"""
    return HyperspectralDataset(
        image_dir="Placenta P007 - P053 red blue/",
        label_dir="Placenta P007 - P053 labels/",
        augmentation=augmentation,
        augmentation_ratio=augmentation_ratio,
        padding_value=padding_value
    )

def get_splits(dataset, k=5):
    """Genera splits para K-Fold Cross Validation"""
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    fold_size = len(dataset) // k
    folds = [(indices[i*fold_size:(i+1)*fold_size], indices[:i*fold_size] + indices[(i+1)*fold_size:]) for i in range(k)]
    return folds