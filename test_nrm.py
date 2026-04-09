#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# bdtools
from bdtools.norm import norm_pct, norm_gcn

# function(s)
from functions import norm_pct_8bits

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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    pvw_paths = list(data_path.rglob("preview.tif"))
    
    # Parameters
    channel = [5, 6, 7]
    
    # Load previews
    pvws = []
    for path in pvw_paths:
        pvw = io.imread(path)[channel, ...]
        pvw = norm_pct_8bits(pvw)
        pvws.append(pvw)
        
    # Extract channel(s)
    imgs = []
    for pvw in pvws:
        img = np.mean(pvw, axis=0)
        img = norm_pct(img, pct_low=0.1, pct_high=99.9)
        imgs.append(img)
    
    # Pad
    imgs = pad_images(imgs)
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        np.stack(imgs), name="imgs", visible=1,
        blending="additive",
        gamma=1.0,
        )
        
    # -------------------------------------------------------------------------

    # imgs_norm0 = []
    # for i, (img, msk) in enumerate(zip(imgs, msks)):
    #     img = norm_pct(img, mask=msk)
    #     imgs_norm0.append(img)
    
    # imgs_norm1 = []
    # for i, (img, msk) in enumerate(zip(imgs, msks)):
    #     img = norm_gcn(img, mask=msk)
    #     img = norm_pct(img, mask=msk)
    #     imgs_norm1.append(img)

    # -------------------------------------------------------------------------

    # imgs_merged = [
    #     np.hstack((img_norm0, img_norm1)) 
    #     for img_norm0, img_norm1 in zip(imgs_norm0, imgs_norm1)
    #     ]
    # msks_merged = [
    #     np.hstack((msk, msk)) for msk in msks
    #     ]
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     np.stack(imgs_merged), name="imgs", visible=1,
    #     blending="additive",
    #     gamma=0.5,
    #     )
    # vwr.add_image(
    #     np.stack(msks_merged), name="msks", visible=1,
    #     blending="additive", colormap="yellow",
    #     opacity=0.2,
    #     )
