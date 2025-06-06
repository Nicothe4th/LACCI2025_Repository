# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:09:11 2025

@author: nicolas
"""

import torch
import time
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset



## Runs one model at time, uncomment the corresponting 
from dataset import get_dataset, get_splits
# from dataset_to_pca import get_dataset, get_splits
# from dataset_to_MSI import get_dataset, get_splits
##

from model import UNet, get_class_weights, get_weighted_loss
from metrics import MetricsEvaluator
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler  # Mixed Precision Training

def train_unet(dataset, num_c, model_name, k=5, epochs=50, batch_size=2):
    splits = get_splits(dataset, k)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    computation_times = []
    fold_metrics = {"Loss": [], "Accuracy": [], "IoU": [], "F1-score": []}
    all_epochs_data = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{k} ---")

        train_subset = get_dataset(augmentation=False)
        val_subset = get_dataset(augmentation=False)

        train_loader = DataLoader(Subset(train_subset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(Subset(val_subset, val_idx), batch_size=batch_size, num_workers=4, pin_memory=True)

        model = UNet(num_channels=num_c, num_classes=7).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scaler = GradScaler()

        all_labels = torch.cat([labels for _, labels in train_loader], dim=0)
        class_weights = get_class_weights(all_labels, num_classes=7).to(device)

        start_time = time.time()
        epoch_metrics = {
            "train_Accuracy": [], "val_Accuracy": [],
            "train_IoU": [], "val_IoU": [],
            "train_F1-score": [], "val_F1-score": []
        }
        epoch_loss = {"train_Loss": []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            train_evaluator = MetricsEvaluator(num_classes=7, device=device)
            epoch_start_time = time.time()

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with autocast("cuda"):
                        if torch.isnan(images).any() or torch.isinf(images).any():
                            raise RuntimeError("NaNs or Infs in input images!")
                        outputs = model(images)
                        loss = get_weighted_loss(outputs, labels, class_weights)
                        if torch.isnan(outputs).any():
                            print("outputs stats:", outputs.min(), outputs.max(), outputs.mean())
                            raise RuntimeError("NaNs in model outputs")
                            
                        loss = get_weighted_loss(outputs, labels, class_weights)
                        if torch.isnan(loss):
                            print("Labels unique:", torch.unique(labels))
                            print("Class weights:", class_weights)
                            raise RuntimeError(" NaN in loss calculation")
                            continue  # Skip this batch
                        

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()
                    train_evaluator.update(outputs.detach(), labels)
                    pbar.set_postfix({"Loss": total_loss / (pbar.n + 1)})

            avg_train_loss = total_loss / len(train_loader)
            train_metrics = train_evaluator.compute()
            epoch_loss["train_Loss"].append(avg_train_loss)
            epoch_metrics["train_Accuracy"].append(train_metrics["Accuracy"])
            epoch_metrics["train_IoU"].append(train_metrics["IoU"])
            epoch_metrics["train_F1-score"].append(train_metrics["F1-score"])

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Accuracy: {train_metrics['Accuracy']:.4f} | IoU: {train_metrics['IoU']:.4f} | F1: {train_metrics['F1-score']:.4f}")

            # Validation
            model.eval()
            val_evaluator = MetricsEvaluator(num_classes=7, device=device)

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_evaluator.update(outputs, labels)

            val_metrics = val_evaluator.compute()
            epoch_metrics["val_Accuracy"].append(val_metrics["Accuracy"])
            epoch_metrics["val_IoU"].append(val_metrics["IoU"])
            epoch_metrics["val_F1-score"].append(val_metrics["F1-score"])

            print(f"Validation - Accuracy: {val_metrics['Accuracy']:.4f} | IoU: {val_metrics['IoU']:.4f} | F1: {val_metrics['F1-score']:.4f}")
            remaining_time = (epochs - epoch - 1) * (time.time() - epoch_start_time)
            print(f"Tiempo restante estimado: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

            # Save metrics to list
            all_epochs_data.append({
                "Model": model_name,
                "Fold": fold + 1,
                "Epoch": epoch + 1,
                "Train_Loss": avg_train_loss,
                "Train_Accuracy": train_metrics["Accuracy"],
                "Val_Accuracy": val_metrics["Accuracy"],
                "Train_IoU": train_metrics["IoU"],
                "Val_IoU": val_metrics["IoU"],
                "Train_F1": train_metrics["F1-score"],
                "Val_F1": val_metrics["F1-score"]
            })

        computation_times.append(time.time() - start_time)

        fold_metrics["Loss"].append(avg_train_loss)
        fold_metrics["Accuracy"].append(val_metrics["Accuracy"])
        fold_metrics["IoU"].append(val_metrics["IoU"])
        fold_metrics["F1-score"].append(val_metrics["F1-score"])

        # Plot Train vs Validation
        plt.figure(figsize=(12, 10))
        for i, metric in enumerate(["Accuracy", "IoU", "F1-score"]):
            plt.subplot(2, 2, i+1)
            plt.plot(epoch_metrics[f"train_{metric}"], label="Train")
            plt.plot(epoch_metrics[f"val_{metric}"], label="Validation")
            plt.title(metric)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(epoch_loss["train_Loss"], label="Train Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/{model_name}_metrics_fold{fold+1}.png")
        plt.show()

        torch.save(model.state_dict(), f"{model_name}_fold{fold+1}.pth")

    # Save all fold/epoch data to CSV
    df = pd.DataFrame(all_epochs_data)
    csv_path = f"results/{model_name}_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    dataset = get_dataset(augmentation=False)
    ##Run one at time, import the corresponding dataset
    train_unet(dataset, num_c=8, model_name='unet')
    # train_unet(dataset, num_c=26, model_name='PCA_unet')
    # train_unet(dataset, num_c=37, model_name='MSI_unet')