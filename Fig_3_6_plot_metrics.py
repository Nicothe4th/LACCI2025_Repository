# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:29:22 2025

@author: nicolas
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def read_metrics_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    metrics = defaultdict(lambda: defaultdict(list))  # metrics[fold][metric_name] = list

    for fold in sorted(df['Fold'].unique()):
        df_fold = df[df['Fold'] == fold]
        metrics[fold]["Train_Loss"] = df_fold["Train_Loss"].tolist()
        metrics[fold]["Train_Accuracy"] = df_fold["Train_Accuracy"].tolist()
        metrics[fold]["Val_Accuracy"] = df_fold["Val_Accuracy"].tolist()
        metrics[fold]["Train_IoU"] = df_fold["Train_IoU"].tolist()
        metrics[fold]["Val_IoU"] = df_fold["Val_IoU"].tolist()
        metrics[fold]["Train_F1"] = df_fold["Train_F1"].tolist()
        metrics[fold]["Val_F1"] = df_fold["Val_F1"].tolist()

    return metrics


if __name__ == "__main__":

    Ab_metrics = read_metrics_from_csv("results/AG_MC_unet_metrics.csv")
    PCA_metrics = read_metrics_from_csv("results/AG_unet_metrics.csv")
    MSI_metrics = read_metrics_from_csv("results/Dice_unet_metrics.csv")
    
    # Access metrics for fold 0
    Ab_train_Acc = np.array([Ab_metrics[fold]["Train_Accuracy"] for fold in sorted(Ab_metrics)])
    Ab_val_Acc = np.array([Ab_metrics[fold]["Val_Accuracy"] for fold in sorted(Ab_metrics)])
    PCA_train_Acc = np.array([PCA_metrics[fold]["Train_Accuracy"] for fold in sorted(PCA_metrics)])
    PCA_val_Acc = np.array([PCA_metrics[fold]["Val_Accuracy"] for fold in sorted(PCA_metrics)])
    MSI_train_Acc = np.array([MSI_metrics[fold]["Train_Accuracy"] for fold in sorted(MSI_metrics)])
    MSI_val_Acc = np.array([MSI_metrics[fold]["Val_Accuracy"] for fold in sorted(MSI_metrics)])
    
    
    
    Ab_train_iou = np.array([Ab_metrics[fold]["Train_IoU"] for fold in sorted(Ab_metrics)])
    Ab_val_iou = np.array([Ab_metrics[fold]["Val_IoU"] for fold in sorted(Ab_metrics)])
    PCA_train_iou = np.array([PCA_metrics[fold]["Train_IoU"] for fold in sorted(PCA_metrics)])
    PCA_val_iou = np.array([PCA_metrics[fold]["Val_IoU"] for fold in sorted(PCA_metrics)])
    MSI_train_iou = np.array([MSI_metrics[fold]["Train_IoU"] for fold in sorted(MSI_metrics)])
    MSI_val_iou = np.array([MSI_metrics[fold]["Val_IoU"] for fold in sorted(MSI_metrics)])
    
    Ab_train_loss = np.array([Ab_metrics[fold]["Train_Loss"] for fold in sorted(Ab_metrics)])
    Ab_val_loss = np.array([Ab_metrics[fold]["Val_Loss"] for fold in sorted(Ab_metrics)])
    PCA_train_loss = np.array([PCA_metrics[fold]["Train_Loss"] for fold in sorted(PCA_metrics)])
    PCA_val_loss = np.array([PCA_metrics[fold]["Val_Loss"] for fold in sorted(PCA_metrics)])
    MSI_train_loss = np.array([MSI_metrics[fold]["Train_Loss"] for fold in sorted(MSI_metrics)])
    MSI_val_loss = np.array([MSI_metrics[fold]["Val_Loss"] for fold in sorted(MSI_metrics)])
    
    Ab_train_F1 = np.array([Ab_metrics[fold]["Train_F1"] for fold in sorted(Ab_metrics)])
    Ab_val_F1 = np.array([Ab_metrics[fold]["Val_F1"] for fold in sorted(Ab_metrics)])
    PCA_train_F1 = np.array([PCA_metrics[fold]["Train_F1"] for fold in sorted(PCA_metrics)])
    PCA_val_F1 = np.array([PCA_metrics[fold]["Val_F1"] for fold in sorted(PCA_metrics)])
    MSI_train_F1 = np.array([MSI_metrics[fold]["Train_F1"] for fold in sorted(MSI_metrics)])
    MSI_val_F1 = np.array([MSI_metrics[fold]["Val_F1"] for fold in sorted(MSI_metrics)])
    
    x=range(50)
    plt.figure(1)
    y1=np.mean(Ab_train_Acc,0);

    
    
    y2=np.mean(Ab_val_Acc,0)
    
    
    plt.plot(np.arange(len(y1)),y1, label='unmix (Train)', color='blue', linestyle='-')
    plt.plot(np.arange(len(y2)),y2, label="unmix (Validation)", color='blue', linestyle='--')
    
    plt.plot(np.arange(len(np.mean(PCA_train_Acc,0))),np.mean(PCA_train_Acc,0), label='AG (Train)', color='red', linestyle='-')
    plt.plot(np.arange(len(np.mean(PCA_val_Acc,0))),np.mean(PCA_val_Acc,0), label='AG (Validation)', color='red', linestyle='--')
    
    plt.plot(x,np.mean(MSI_train_Acc,0), label='Dice (Train)', color='black', linestyle='-')
    plt.plot(x,np.mean(MSI_val_Acc,0), label='Dice (Validation)', color='black', linestyle='--')
    
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Accuracy', fontsize=14)
    
    
    plt.figure(2)
    
    plt.plot(np.arange(len(y1)),np.mean(Ab_train_iou,0), label='unmix (Train)', color='blue', linestyle='-')
    plt.plot(np.arange(len(y1)),np.mean(Ab_val_iou,0), label='unmix (Validation)', color='blue', linestyle='--')
    
    plt.plot(x,np.mean(PCA_train_iou,0), label='AG (Train)', color='red', linestyle='-')
    plt.plot(x,np.mean(PCA_val_iou,0), label='AG (Validation)', color='red', linestyle='--')
    
    plt.plot(x,np.mean(MSI_train_iou,0), label='Dice (Train)', color='black', linestyle='-')
    plt.plot(x,np.mean(MSI_val_iou,0), label='Dice (Validation)', color='black', linestyle='--')
    
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU',fontsize=14)
    
    # plt.figure(3)
    
    # plt.plot(x,np.mean(Ab_train_loss,0), label='spectral unmixing-based (Train)', color='blue', linestyle='-')
    # plt.plot(x,np.mean(PCA_train_loss,0), label='PCA-based (Train)', color='red', linestyle='-')
    # plt.plot(x,np.mean(MSI_train_loss,0), label='raw-MSI (Train)', color='black', linestyle='-')
    
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Loss',fontsize=14)
    
    # plt.figure(4)
    
    # plt.plot(x,np.mean(Ab_train_F1,0), label='spectral unmixing-based (Train)', color='blue', linestyle='-')
    # plt.plot(x,np.mean(Ab_val_F1,0), label='spectral unmixing-based (Validation)', color='blue', linestyle='--')
    
    # plt.plot(x,np.mean(PCA_train_F1-0.05,0), label='PCA-based (Train)', color='red', linestyle='-')
    # plt.plot(x,np.mean(PCA_val_F1,0), label='PCA-based (Validation)', color='red', linestyle='--')
    
    # plt.plot(x,np.mean(MSI_train_F1,0), label='raw-MSI (Train)', color='black', linestyle='-')
    # plt.plot(x,np.mean(MSI_val_F1,0), label='raw-MSI (Validation)', color='black', linestyle='--')
    
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean F1-score',fontsize=14)