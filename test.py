#%% Imports -------------------------------------------------------------------

import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# function(s)
from functions import get_intensities

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
    pvw_raw_paths = list(data_path.rglob("preview_raw.tif"))
    pvw_msk_paths = list(data_path.rglob("preview_mask.tif"))
    
    # Parameters
    channel = 5
    
    # Load data
    pvws, msks = [], []
    for pvw_path, msk_path in zip(pvw_raw_paths, pvw_msk_paths):
        pvw = io.imread(pvw_path)[channel]
        msk = io.imread(msk_path)
        pvws.append(pvw)
        msks.append(msk)
        
    # Pad
    pvws = pad_images(pvws)
    msks = pad_images(msks)
        
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
    
#%%

    def save_bit_packing(arr, path):
        arr_bin = np.packbits(arr)
        np.savez_compressed(path, data=arr_bin, org_shape=arr.shape)
        
    def load_bit_packing(path):
        loader = np.load(path)
        arr_bin = loader["data"]
        shape = loader["org_shape"]
        unpacked = np.unpackbits(arr_bin)[:np.prod(shape)]
        return unpacked.reshape(shape) * 255

    # Paths
    vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    dap_msk_path = vws_path / "dapi_mask.tif"
    dap_msk_bin_path = vws_path / "dapi_mask.npz"
    
    # Load
    dap_msk = io.imread(dap_msk_path)
           
    # Save bit-packing
    save_bit_packing(dap_msk, dap_msk_bin_path)
    
    # Load bit-packing
    dap_msk_bin = load_bit_packing(dap_msk_bin_path)
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        dap_msk_bin, name="dap_msk", visible=1,
        blending="additive",
        )
    
    
    
