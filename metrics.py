#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:18:54 2025

@author: jnmc
"""

import torchmetrics
import torch

# class MetricsEvaluator:
#     def __init__(self, num_classes, device):
#         self.num_classes = num_classes
#         self.device = device

#         self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
#         self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro").to(device)
#         self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

#     def reset(self):
#         self.accuracy.reset()
#         self.iou.reset()
#         self.f1.reset()

#     def update(self, preds, targets):
#         preds = torch.argmax(preds, dim=1)  # Convertir logits a clases
#         self.accuracy.update(preds, targets)
#         self.iou.update(preds, targets)
#         self.f1.update(preds, targets)

#     def compute(self):
#         return {
#             "Accuracy": self.accuracy.compute().item(),
#             "IoU": self.iou.compute().item(),
#             "F1-score": self.f1.compute().item()
#         }

class MetricsEvaluator:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

        # Definir las métricas de evaluación
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    def reset(self):
        """Resetear las métricas."""
        self.accuracy.reset()
        self.iou.reset()
        self.f1.reset()

    def update(self, preds, targets):
        """Actualizar las métricas con las predicciones y las etiquetas."""
        preds = torch.argmax(preds, dim=1)  # Convertir logits a clases
        
        # Actualizar las métricas
        self.accuracy.update(preds, targets)
        self.iou.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self):
        """Calcular y devolver las métricas."""
        return {
            "Accuracy": self.accuracy.compute().item(),
            "IoU": self.iou.compute().item(),
            "F1-score": self.f1.compute().item()
        }