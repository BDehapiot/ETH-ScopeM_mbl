#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

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

def prep_preview(pvw, channels=[2, 5, 6]):
    return np.mean(norm_pct_8bits(pvw)[channels, ...], axis=0).astype("uint8")  

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Paths
    data_path = Path("D:\local_mbl\data")
    pvw_paths = list(data_path.rglob("preview.tif"))
    msk_paths = list(data_path.rglob("mask.tif"))
    
    # Parameters
    channels = [2, 5, 6]
    
#%% Prepare previews ----------------------------------------------------------

    # # Load previews
    # pvws = []
    # for path in pvw_paths:
    #     pvw = io.imread(path)
    #     pvw = prep_preview(pvw, channels=channels)
    #     pvws.append(pvw)
            
    # # Pad
    # pvws = pad_images(pvws)
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     np.stack(pvws), name="pvws", visible=1,
    #     blending="additive",
    #     gamma=1.0,
    #     )
    
#%% Check masks ---------------------------------------------------------------

    # # Load data
    # pvws, msks = [], []
    # for pvw_path, msk_path in zip(pvw_paths, msk_paths):
    #     pvw = io.imread(pvw_path)
    #     msk = io.imread(msk_path)
    #     pvw = prep_preview(pvw, channels=channels)
    #     pvws.append(pvw)
    #     msks.append(msk)
        
    # # Pad
    # pvws = pad_images(pvws)
    # msks = pad_images(msks)
        
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     np.stack(pvws), name="pvws", visible=1,
    #     blending="additive",
    #     gamma=0.75,
    #     )
    # vwr.add_image(
    #     np.stack(msks), name="msks", visible=1,
    #     blending="additive", colormap="yellow",
    #     opacity=0.25,
    #     )

#%% Get stats -----------------------------------------------------------------

    # Load data
    pvws, msks = [], []
    for pvw_path, msk_path in zip(pvw_paths, msk_paths):
        pvw = io.imread(pvw_path)
        msk = io.imread(msk_path)
        pvws.append(pvw)
        msks.append(msk)
        
    # Get statistics
    
    stats = {
        "avg" : [],
        "med" : [],
        "std" : [],
        "qtl" : [],
        }
    
    for pvw, msk in zip(pvws, msks):
        
        for img in pvw:

            stats["avg"].append(np.mean(img[msk]))
            stats["med"].append(np.median(img[msk]))
            stats["std"].append(np.std(img[msk]))
            
            quantiles = []
            for q, qtl in enumerate(np.arange(0.1, 1.0, 0.1)):
                quantiles.append(np.quantile(img[msk], qtl))
            stats["qtl"].append(quantiles)