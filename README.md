# LACCI2025_Repository
This repository contains the scripts used to reproduce the experiments conducted for the LACCI 2025 conference.
The MSI database paper is available in https://doi.org/10.1016/j.dib.2023.109526 by Pusteen at al., and the images are available for download in: 10.5281/zenodo.8045940

Downoload de database and create a folder called: Placenta P007 - P053 red blue, and place the respective MSI images
  1.- make_label_map.py extracts the labels for the images and place it within a folder called Placenta P007 - P053 labels
  2.- unmixing_feature_extraction.py performs the unmixing task, and place the results in Placenta P007 - P053 unmixing
  3.- trainig_stage.py perfomrs the U-net training for database of (i) unmixing, (ii) PCA-based and (iii) raw-MSI, run one databased at time, comenting the non-used import database.
  4.- Fig_3_6_plot_metrics.py plots the results metrics
  5.- Figure7_getlabelmaps.py clasify a image and deploys the labels maps.

