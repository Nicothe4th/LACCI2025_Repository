#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:01:29 2025

@author: jnmc
"""
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from sklearn.model_selection import KFold
import os

class HyperspectralDataset(Dataset):
    def __init__(self, image_dir, label_dir, augmentation=False, augmentation_ratio=0.1, padding_value=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

        assert len(self.image_files) == len(self.label_files), "Número de imágenes y etiquetas no coincide"

        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.padding_value = padding_value

        # Seleccionar aleatoriamente imágenes para aplicar augmentation
        num_augmented = int(len(self.image_files) * self.augmentation_ratio)
        self.augmented_indices = set(random.sample(range(len(self.image_files)), num_augmented))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Cargar imagen y etiquetas
        image = np.load(os.path.join(self.image_dir, self.image_files[idx]))  # (1024, 1024, 9)
        label = np.load(os.path.join(self.label_dir, self.label_files[idx]))  # (1024, 1024)

        # Convertir a tensores de PyTorch
        image = image.T.reshape((1024, 1024, 8), order='F') 
        image = torch.tensor(image, dtype=torch.float32).permute(dims=[2, 0, 1])  
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)  

        # Aplicar data augmentation solo a los índices seleccionados
        if self.augmentation and idx in self.augmented_indices:
            image, label = self.apply_augmentation(image, label)

        return image, label.squeeze(0)  # Quitar la dimensión extra del label al final

    def apply_augmentation(self, image, label):
        """Aplica rotaciones y flips de forma consistente en imagen y etiquetas, con padding reflectante."""
        angle = random.uniform(-15, 15)
        
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
        image_dir="Placenta P007 - P053 unmixing/",
        label_dir="Placenta P007 - P053 labels/",
        augmentation=augmentation,
        augmentation_ratio=augmentation_ratio,
        padding_value=padding_value
    )


def get_splits(dataset, k=5, seed=None, shuffle=True):
    """Genera splits para K-Fold Cross Validation usando sklearn."""
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)

    indices = np.arange(len(dataset))
    splits = [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in kf.split(indices)]
    
    return splits