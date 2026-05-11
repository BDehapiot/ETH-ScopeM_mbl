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

# bdtools
from bdtools.model.model import Model

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

#%% Compressed stack ----------------------------------------------------------

    # # Paths
    # data_path = Path("D:\local_mbl\data") 
    # vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    # mtd_path = data_path / (vws_path.name[:-3] + "_metadata.pkl")
    # ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    # stk_nrm_path = vws_path / "stack_norm.npz"

    # # Parameters
    # crop_size = 32

    # # Load
    # with open(mtd_path, "rb") as file:
    #     mtd = pickle.load(file)
    # with open(ncl_ctd_path, "rb") as file:
    #     ncl_ctd = pickle.load(file)
    # stk_nrm = load_compressed_stack(stk_nrm_path)

    # # -------------------------------------------------------------------------

    # t0 = time.time()
    # print("get_crops() : ", end="", flush=True)

    # # Get crops
    # crp_nrm = get_crops(
    #     stk_nrm, ncl_ctd, 
    #     crop_size=crop_size, 
    #     crop_fraction=1, 
    #     stk_level=1,
    #     )
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

#%% Preview mask --------------------------------------------------------------    

    # # Paths
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
    
#%% Normalization -------------------------------------------------------------
    
    # # Paths
    # pvw_raw_paths = list(data_path.rglob("preview_raw.tif"))
    # pvw_nrm_paths = list(data_path.rglob("preview_norm.tif"))
    
    # # Parameters
    # channel = 4
    
    # # Load
    # pvw_raws, pvw_nrms = [], []
    # for pvw_raw_path, pvw_nrm_path in zip(pvw_raw_paths, pvw_nrm_paths):
    #     pvw_raw = io.imread(pvw_raw_path)[channel]
    #     pvw_nrm = io.imread(pvw_nrm_path)[channel]
    #     pvw_raws.append(pvw_raw)
    #     pvw_nrms.append(pvw_nrm)    
        
    # # Pad
    # pvw_raws = pad_images(pvw_raws)
    # pvw_nrms = pad_images(pvw_nrms)
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     np.stack(pvw_raws), name="raws", visible=1,
    #     blending="additive", colormap="inferno",
    #     gamma=0.35,
    #     )
    # vwr.add_image(
    #     np.stack(pvw_nrms), name="nrms", visible=1,
    #     blending="additive", colormap="inferno",
    #     gamma=0.35,
    #     )  
    # vwr.grid.enabled = True
    
#%% Crops ---------------------------------------------------------------------

    # Paths
    vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    stk_nrm_path = vws_path / "stack_norm.npz"
    ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    model_aec_path = Path.cwd() / "model-aec_32_3_128_59273-None"
    
    # Parameters
    crop_size = 32
    crop_fraction = 0.05
    stk_level = 1
    
    # Load
    stk_nrm = load_compressed_stack(stk_nrm_path)
    with open(ncl_ctd_path, "rb") as file:
        ncl_ctd = pickle.load(file)
        
    # Load model
    model_aec = Model(parameters=None, model_path=model_aec_path)
        
    # Get crops
    crp_nrm = get_crops(
        stk_nrm, ncl_ctd, 
        crop_size=crop_size, 
        crop_fraction=crop_fraction, 
        stk_level=stk_level,
        )
    
    # Prepare crops
    crp_nrm = crp_nrm[:, :7, ...] # remove autofluo channel
    crp_nrm = np.transpose(crp_nrm, (0, 2, 3, 1)) # transposition
    crp_nrm = crp_nrm.astype("float32") / 255
    
    # Predict (aec)
    prd = model_aec.predict(
        crp_nrm, 
        patch_overlap=0, 
        batch_size=32, 
        chunk_size=None, 
        latent=False
        )
    
    # Predict (latent space)
    lsp = model_aec.predict(
        crp_nrm, 
        patch_overlap=0, 
        batch_size=32, 
        chunk_size=10000, 
        latent=True
        )
    
    # Transpose
    crp_nrm = np.transpose(crp_nrm, (0, 3, 1, 2))
    prd = np.transpose(prd, (0, 3, 1, 2))
    
    # Display (aec)
    vwr = napari.Viewer()
    vwr.add_image(
        np.stack(crp_nrm), name="crp_nrm", visible=1,
        blending="additive", colormap="inferno",
        gamma=0.75,
        )
    vwr.add_image(
        np.stack(prd), name="prd", visible=1,
        blending="additive", colormap="inferno",
        gamma=0.75,
        )
    vwr.grid.enabled = True
    
    # Display (latent space)
    vwr = napari.Viewer()
    vwr.add_image(
        np.stack(lsp), name="lsp", visible=1,
        blending="additive", colormap="inferno",
        gamma=1,
        )
