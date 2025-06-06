#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:28:16 2025

@author: jnmc
"""

import os
import sys
import re
import numpy as np
from spectral_tiffs import read_stiff  
sys.path.insert(0, "/Users/nicol/Documents/unmixing algorithms/python")
from NEBEAETV import nebeaetv
from NEBEAE import nebeae

# Define folder paths
base_folder = "/Users/nicol/Documents/placenta"
input_folder = os.path.join(base_folder, "Placenta P007 - P053 red blue")
output_folder = os.path.join(base_folder, "Placenta P007 - P053 unmixing")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Regular expression to match 'PXXX.tif' (XXX is a number)
pattern = re.compile(r"P\d{3}\.tif$")

# List only valid 'PXXX.tif' files
tif_files = sorted([f for f in os.listdir(input_folder) if pattern.match(f)])

for tif_file in tif_files:
    image_path = os.path.join(input_folder, tif_file)
    
    # Extract the patient ID (e.g., 'P015' from 'P015.tif')
    patient_id = tif_file.split('.')[0]

    # Read the spectral image
    imageC, center_wavelengths, image_rgbC, metadata = read_stiff(image_path)
    
    # Load the corresponding mask
    mask_path = image_path.replace(".tif", "_labeled.npy")
    mask_path = mask_path.replace("red blue","labels")
    if not os.path.exists(mask_path):
        print(f"Skipping {tif_file}: Mask file not found.")
        continue  # Skip this file if the mask is missing
    
    masks = np.load(mask_path)
    M = masks.reshape((-1, 1), order='F')

    # Image dimensions
    Ny, Nx, Nz = imageC.shape
    Z = imageC.reshape((Nx * Ny, Nz), order='F').T
    Z = Z / np.sum(Z, axis=0, keepdims=True)  # Normalize

    L, K = Z.shape  # L = spectral bands, K = pixels

    labels = range(1, 7)  # Labels from 1 to 7

    # Dictionaries to store indices, Z arrays, and P arrays
    I_dict = {}
    Z_dict = {}
    P_dict = {}

    # 1.- Artery
    I_dict[1], _ = np.where(M == 1)  # Find indices
    if I_dict[1].size > 0:  # Only process if there are valid indices
        Z_dict[1] = Z[:, I_dict[1]].copy()
        P_dict[1] = np.mean(Z_dict[1], axis=1, keepdims=True)
    else:
        Z_dict[1] = None  # No valid data
        P_dict[1] = None
        
    # 2 y 3 .- Specular reflection
    I_dict[2], _ = np.where(M == 2)  # Find indices
    if I_dict[2].size > 0:  # Only process if there are valid indices
        Z_dict[2] = Z[:, I_dict[2]].copy()
        par = [6, 0.1, 0.1, 1e-3,20, 0, 1, 0]
        P_sr,_,_,_,_,_= nebeae(Z_dict[2],2,par)
        P_dict[2] = P_sr[:,0].reshape((L,1))
        P_dict[3] = P_sr[:,1].reshape((L,1))
    else:
        Z_dict[2] = None  # No valid data
        P_dict[2] = None
        P_dict[3] = None
        
    # 4.- Stroma
    I_dict[4], _ = np.where(M == 3)  # Find indices
    if I_dict[4].size > 0:  # Only process if there are valid indices
        Z_dict[4] = Z[:, I_dict[4]].copy()
        P_dict[4] = np.mean(Z_dict[4], axis=1, keepdims=True)
    else:
        Z_dict[4] = None  # No valid data
        P_dict[4] = None
    # 5.- Vein
    I_dict[5], _ = np.where(M == 4)  # Find indices
    if I_dict[5].size > 0:  # Only process if there are valid indices
        Z_dict[5] = Z[:, I_dict[5]].copy()
        P_dict[5] = np.mean(Z_dict[5], axis=1, keepdims=True)
    else:
        Z_dict[5] = None  # No valid data
        P_dict[5] = None
        
    # 6.- Suture
    I_dict[6], _ = np.where(M == 5)  # Find indices
    if I_dict[6].size > 0:  # Only process if there are valid indices
        Z_dict[6] = Z[:, I_dict[6]].copy()
        P_dict[6] = np.mean(Z_dict[6], axis=1, keepdims=True)
    else:
        Z_dict[6] = None  # No valid data
        P_dict[6] = None
    # 7.- Umbilicar Cord
    I_dict[7], _ = np.where(M == 6)  # Find indices
    if I_dict[7].size > 0:  # Only process if there are valid indices
        Z_dict[7] = Z[:, I_dict[7]].copy()
        P_dict[7] = np.mean(Z_dict[7], axis=1, keepdims=True)
    else:
        Z_dict[7] = None  # No valid data
        P_dict[7] = None

    # Create list of valid P arrays and their indices
    valid_P = [P for P in P_dict.values() if P is not None]
    valid_indices = [label for label, P in P_dict.items() if P is not None]

    # Concatenate valid P arrays
    P_final = np.concatenate(valid_P, axis=1) if valid_P else np.array([])
    P = P_final / np.sum(P_final, axis=0, keepdims=True) if P_final.size > 0 else np.array([])

    # Convert valid indices to NumPy array
    valid_indices = np.array(valid_indices)
    _, N = P_final.shape if P_final.size > 0 else (0, 0)

    # Skip processing if no valid data
    if N == 0:
        print(f"Skipping {tif_file}: No valid end-members found.")
        continue

    # Output results
    print(f"Processing {tif_file}:")
    print("Concatenated array (P_final) shape:", P_final.shape)
    print("Indices of valid P arrays:", valid_indices)

    # Parameters for NEBEAE-TV algorithm
    initcond = 6  # Initial condition of end-members matrix: 6 (VCA) and 8 (SISAL)
    rho = 0.1  # Similarity weight in end-members estimation
    Lambda = 0.1  # Entropy weight for abundance estimation
    epsilon = 1e-3
    maxiter = 20
    parallel = 1
    downsampling = 0.0  # Downsampling in end-members estimation
    display_iter = 0  # Display partial performance in BEAE
    lm = 0.1

    partv = [initcond, rho, 1e-4, 0.1, 10, Ny, Nx, epsilon, maxiter, parallel, display_iter]
    Ptv, Atv, Wtv, Dstv, Stv, Yhtv, Jitv = nebeaetv(Z, N, partv, P, 1)

    # Initialize feature_vectors array (9 rows, K columns)
    feature_vectors = np.zeros((8, K))

    # Fill feature_vectors with abundance values in order of end-member labels
    for i, idx in enumerate(valid_indices):
        feature_vectors[idx - 1, :] = Atv[i, :].copy()

    # Store the distance metric in the last row
    feature_vectors[7, :] = Dstv.copy()

    # Save the feature vector in the output folder
    save_path = os.path.join(output_folder, f"{patient_id}_feature.npy")
    np.save(save_path, feature_vectors)
    print(f"Saved feature vector: {save_path}")

print("Processing complete!")
