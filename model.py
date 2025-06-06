#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:02:46 2025

@author: jnmc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast  # Mixed Precision Training

def safe_cat(a, b, name):
    if torch.isnan(a).any() or torch.isnan(b).any():
        print(f" NaN detected before cat in {name}")
        print('a=',a)
        print('b=',b)
        raise RuntimeError("NaN encountered during forward pass — aborting.")
    return torch.cat([a, b], 1)


def double_conv(in_c, out_c):
    """Dos capas convolucionales + BatchNorm + ReLU, con padding para no recortar tamaño."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),    # Normaliza después de la 1a convolución
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),    # Normaliza después de la 2a convolución
        nn.ReLU(inplace=True)
    )

def double_convD(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, num_channels=8, num_classes=7):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.down_conv_1 = double_conv(num_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        # Decoder
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, image):
        """Forward con Mixed Precision para reducir consumo de memoria."""
        with autocast("cuda"):  # Usa float16 en GPUs compatibles
            # Encoder
            x1 = self.down_conv_1(image)  
            if torch.isnan(x1).any(): print("NaN in down_conv_1")
            
            x2 = self.max_pool_2x2(x1)
            x3 = self.down_conv_2(x2)
            if torch.isnan(x2).any(): print("NaN in down_conv_2")
            
            x4 = self.max_pool_2x2(x3)
            x5 = self.down_conv_3(x4)
            if torch.isnan(x5).any(): print("NaN in down_conv_3")
            
            x6 = self.max_pool_2x2(x5)
            x7 = self.down_conv_4(x6)
            if torch.isnan(x7).any(): print("NaN in down_conv_4")
            
            x8 = self.max_pool_2x2(x7)
            x9 = self.down_conv_5(x8)
            if torch.isnan(x9).any(): print("NaN in down_conv_5")

            # Decoder
            x = self.up_trans_1(x9.float())
            x = self.up_conv_1(safe_cat(x, x7.float(),"up_conv_1") )
            if torch.isnan(x).any(): print("NaN in up_conv_1")
            

            x = self.up_trans_2(x.float())
            x = self.up_conv_2(safe_cat(x, x5.float(),"up_conv_2") )
            if torch.isnan(x).any(): print("NaN in up_conv_2")

            x = self.up_trans_3(x.float())
            x = self.up_conv_3(safe_cat(x, x3.float(),"up_conv_3") )
            if torch.isnan(x).any(): print("NaN in up_conv_3")

            x = self.up_trans_4(x.float())
            x = self.up_conv_4(safe_cat(x, x1.float(),"up_conv_4") )
            if torch.isnan(x).any(): print("NaN in up_conv_4")

            x = self.out(x)

        return x


def get_class_weights(labels, num_classes=7, max_weight=1.0):
    flat_labels = labels.flatten()
    class_counts = torch.bincount(flat_labels, minlength=num_classes).float()

    # Reemplaza conteos cero con 1 para evitar pesos infinitos
    class_counts[class_counts == 0] = 1.0

    # Invertir proporcionalmente
    class_weights = 1.0 / class_counts

    # Limita pesos extremos para evitar explosiones numéricas
    class_weights = torch.clamp(class_weights, max=max_weight)

    # Normaliza
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights.to(labels.device)

def get_weighted_loss(model_output, target, class_weights):
    """Calcula la pérdida con los pesos de clase, incluyendo la clase 0 (fondo)."""
    # Asegurarse de que los pesos de clase estén en el mismo dispositivo que el modelo
    class_weights = class_weights.to(model_output.device).to(model_output.dtype)

    # Crear la función de pérdida con los pesos de clase
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    loss = criterion(model_output, target.long())

    return loss
