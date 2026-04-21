#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# function(s)
from functions import (
    load_view, norm_12_to_8bits,
    save_compressed_stack, load_compressed_stack,
    get_crops,
    )

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

    # # Paths
    # data_path = Path("D:\local_mbl\data")
    # pvw_raw_paths = list(data_path.rglob("preview_raw.tif"))
    # pvw_msk_paths = list(data_path.rglob("preview_mask.tif"))
    
    # # Parameters
    # channel = 5
    
    # # Load
    # pvws, msks = [], []
    # for pvw_path, msk_path in zip(pvw_raw_paths, pvw_msk_paths):
    #     pvw = io.imread(pvw_path)[channel]
    #     msk = io.imread(msk_path)
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
    
#%% Compressed stack ----------------------------------------------------------

    # # Paths
    # data_path = Path("D:\local_mbl\data")    
    # vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    # qpt_path = data_path / (vws_path.name[:-3] + ".qptiff")
    # mtd_path = data_path / (vws_path.name[:-3] + "_metadata.pkl")
    # roi_idx  = int(vws_path.name.split("_")[-1])
    # stk_raw_path = vws_path / "stack_raw.npz"
    
    # # Parameters
    # prc_level = 1
    
    # # Load #1
    # with open(mtd_path, "rb") as file:
    #     mtd = pickle.load(file)
    # stk_raw = load_view(
    #     qpt_path, metadata=mtd, channels="all", 
    #     level=prc_level, roi_idx=roi_idx
    #     )
    # stk_raw = norm_12_to_8bits(stk_raw)
    
    # # Save
    # t0 = time.time()
    # print("save_compressed_stack() : ", end="", flush=True)
    # save_compressed_stack(stk_raw, stk_raw_path)
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Load #2
    # t0 = time.time()
    # print("load_compressed_stack() : ", end="", flush=True)
    # stk_raw_loaded = load_compressed_stack(stk_raw_path)
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     stk_raw_loaded, name="stk_raw", visible=1,
    #     blending="additive",
    #     )
    
#%% get crops -----------------------------------------------------------------

    # Paths
    data_path = Path("D:\local_mbl\data")    
    vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    stk_raw_path = vws_path / "stack_raw.npz"
    
    # Parameters
    crop_size = 32
    crop_fraction = 0.1
    stk_level = 1
    
    # Load ncl_ctd
    with open(ncl_ctd_path, "rb") as file:
        ncl_ctd = pickle.load(file)
    
    # Load stk_raw
    t0 = time.time()
    print("load_compressed_stack() : ", end="", flush=True)
    stk_raw = load_compressed_stack(stk_raw_path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # get_crops()
    t0 = time.time()
    print("get_crops() : ", end="", flush=True)
    crops = get_crops(
        stk_raw, ncl_ctd, 
        crop_size=crop_size, 
        crop_fraction=crop_fraction, 
        stk_level=stk_level,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
#%% Zarr stack ----------------------------------------------------------------

    # import zarr

    # # Paths
    # data_path = Path("D:\local_mbl\data")    
    # vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    # qpt_path = data_path / (vws_path.name[:-3] + ".qptiff")
    # mtd_path = data_path / (vws_path.name[:-3] + "_metadata.pkl")
    # roi_idx  = int(vws_path.name.split("_")[-1])
    # stk_raw_path = vws_path / "stack_raw.zarr"
    
    # # Parameters
    # prc_level = 1
    
    # # Load #1
    # with open(mtd_path, "rb") as file:
    #     mtd = pickle.load(file)
    # stk_raw = load_view(
    #     qpt_path, metadata=mtd, channels="all", 
    #     level=prc_level, roi_idx=roi_idx
    #     )
    # stk_raw = norm_12_to_8bits(stk_raw)
    
    # # Save
    # t0 = time.time()
    # print("save zarr : ", end="", flush=True)
    # zarr.save(str(stk_raw_path), stk_raw, chunks=True)
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Load
    # t0 = time.time()
    # print("load zarr : ", end="", flush=True)
    # stk_raw_loaded = zarr.load(str(stk_raw_path))
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
        
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     stk_raw_loaded, name="stk_raw", visible=1,
    #     blending="additive",
    #     )