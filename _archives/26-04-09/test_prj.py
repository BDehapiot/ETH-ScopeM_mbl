#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# bdtools
from bdtools.norm import norm_pct, norm_gcn

#%% Function(s) ---------------------------------------------------------------

def pad_images(imgs):
    imgs_padded = []
    max_y = np.max([img.shape[0] for img in imgs])
    max_x = np.max([img.shape[1] for img in imgs])
    for img in imgs:
        pad_y = max_y - img.shape[0]
        pad_x = max_x - img.shape[1]
        top, bottom = pad_y // 2, pad_y - (pad_y // 2)
        left, right = pad_x // 2, pad_x - (pad_x // 2)
        padding = ((top, bottom), (left, right))
        imgs_padded.append(
            np.pad(img, padding, mode="constant", constant_values=0))
    return imgs_padded

def fetch_measures(mtds, measure, channel):
    measures = []
    for mtd in mtds:
        for idx in range(len(mtd["roi"])):
            measures.append(mtd["stats"][idx][measure][channel])
    return measures

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    mtd_paths = list(data_path.rglob("metadata.pkl"))
    
    # Parameters
    channels = [5, 6]
    
    # Load metadata
    mtds = []
    for path in mtd_paths:
        with open(path, "rb") as file:
            mtds.append(pickle.load(file))
    
    # Extract images
    imgs = []
    for mtd in mtds:
        for idx in range(len(mtd["roi"])):
            img = mtd["preview"][idx][channels, ...]
            img = np.mean(img, axis=0)
            imgs.append(img)
    
    # Pad images
    imgs = pad_images(imgs)
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        np.stack(imgs), name="imgs", visible=1,
        blending="additive",
        gamma=0.5,
        )
    
