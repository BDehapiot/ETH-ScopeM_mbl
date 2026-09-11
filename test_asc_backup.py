#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path
from main import main_parameters

# functions
from functions import load_view, norm_data

# bdtools
from bdtools.model.model import Model

# skimage
from skimage import draw
from skimage.transform import rescale
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops_table

# scipy
from scipy.stats import pearsonr

#%% Inputs --------------------------------------------------------------------

test_parameters = {
    
    # predict_asc()
    "asc_level"  : 0,
    "asc_thresh" : 0.05,
    "min_dist"   : 5,   
    "min_int"    : 0.075, 
    "radius"     : 5, 
    "sigma"      : 1.0, 
    "min_corr"   : 0.8, 
    
    }

main_parameters.update(test_parameters)

#%% Function : get_asc_mask() -------------------------------------------------

def get_asc_mask(
        asc_nrm, asc_prd, asc_thresh=0.05, stk_level=1, asc_level=0):
    
    # Rescale asc predictions
    rf = 1 / (2 ** (stk_level - asc_level))
    asc_prd = rescale(asc_prd, rf)
    
    # Get mask
    asc_msk = asc_prd > asc_thresh
    asc_prc = asc_nrm.copy()
    asc_prc[asc_msk == 0] = 0
    
    return asc_msk, asc_prc, asc_prd

#%% Function : get_asc_speckles() ---------------------------------------------

def get_gaussian_kernel(radius, sigma):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.max()

def get_valid_speckles(
        asc_nrm, pvw_msk, coords, 
        radius=7, sigma=1.0, min_corr=0.9,
        stk_level=1, pvw_level=4,
        ):
    
    # Get gaussian kernel
    kernel = get_gaussian_kernel(radius, sigma)
    kernel_flat = kernel.ravel()

    valid = []
    r = radius
    h, w = asc_nrm.shape
    rf = 1 / (2 ** (pvw_level - stk_level))
    for y, x in coords:
        
        # Check boundaries
        if y - r < 0 or y + r >= h or x - r < 0 or x + r >= w:
            continue
        
        # Check that coords in pvw_msk
        py, px = int(y * rf), int(x * rf)
        if py < 0 or py >= pvw_msk.shape[0] or px < 0 or px >= pvw_msk.shape[1]:
            continue
        if pvw_msk[py, px] == 0:
            continue
                    
        # Patch and correlate
        patch = asc_nrm[y - r : y + r + 1, x - r : x + r + 1]
        corr, _ = pearsonr(patch.ravel(), kernel_flat)
        if corr >= min_corr:
            valid.append([y, x])

    return np.array(valid)

def get_speckle_display(asc_msk, coords, valid_coords, radius):

    asc_dsp = np.zeros_like(asc_nrm, dtype=np.uint8)
    # asc_dsp[asc_msk > 0] = 64
    
    for y, x in coords:
        rr, cc = draw.disk((y, x), 1, shape=asc_msk.shape)
        asc_dsp[rr, cc] = 255
    for y, x in valid_coords:
        rr, cc = draw.circle_perimeter(
            y, x, radius * 2, shape=asc_msk.shape)
        asc_dsp[rr, cc] = 255
                
    return asc_dsp.astype("uint8")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    t0 = time.time()

    # Paths
    data_path = main_parameters["data_path"]
    model_asc_name = main_parameters["model_asc_name"]
    qpt_name = "251126_D7_S2_Scan1_unmixed.qptiff"
    mtd_name = qpt_name.replace(".qptiff", "_metadata.pkl")
    vws_name = qpt_name.replace(".qptiff", "_00")
    qpt_path = data_path / qpt_name
    mtd_path = data_path / mtd_name
    vws_path = data_path / vws_name 

    # -------------------------------------------------------------------------
    
    # Paths
    pvw_sts_path = vws_path / "preview_stats.pkl"
    pvw_msk_path = vws_path / "preview_mask.tif"
    stk_nrm_path = vws_path / "stack_norm.tif"
    asc_msk_path = vws_path / "asc_mask.tif"
    asc_prc_path = vws_path / "asc_processed.tif"
    asc_dsp_path = vws_path / "asc_display.tif"
    roi_idx = int(vws_path.stem[-2:])

    # Log
    t0i = time.time()
    log_str = f"Main() : predict_asc() : {vws_path.name}"
    print("-" * len(log_str))
    print(log_str)
    print("-" * len(log_str))

    # Model
    model_asc_path = Path.cwd() / model_asc_name
    if "model_asc" not in locals():
        model_asc = Model(
            parameters=None, model_path=model_asc_path)
                    
    # Load --------------------------------------------------------------------
    
    t0 = time.time()
    
    with open(mtd_path, "rb") as file:
        mtd = pickle.load(file)
    with open(pvw_sts_path, "rb") as file:
        pvw_sts = pickle.load(file)
    pvw_msk = io.imread(pvw_msk_path)
    stk_nrm = io.imread(stk_nrm_path)
    asc_raw = load_view(
        qpt_path, metadata=mtd, channels="ASC", 
        level=main_parameters["asc_level"], roi_idx=roi_idx,
        )
    
    asc_nrm = norm_data(asc_raw, pvw_sts, channel="ASC")
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s - load")
    
    # Predict -----------------------------------------------------------------
    
    t0 = time.time()
    
    asc_prd = model_asc.predict(
        asc_nrm, patch_overlap=32, batch_size=32, chunk_size=5000)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s - predict")
    
    # Get mask ----------------------------------------------------------------
    
    t0 = time.time()
    
    asc_nrm = stk_nrm[4, ...].astype("float32") / 255
    asc_msk, asc_prc, asc_prd = get_asc_mask(
        asc_nrm, asc_prd, main_parameters["asc_thresh"])
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s - get mask")
    
    # Get speckles ------------------------------------------------------------
    
    t0 = time.time()
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s - get speckles")
    
    # -------------------------------------------------------------------------
    
    t1f = time.time()
    print(f"> {t1f - t0i:.3f}s")
                                       
#%%

    # Process asc mask
    asc_msk = remove_small_objects(asc_msk, min_size=4)
    spk_msk = asc_msk != remove_small_objects(asc_msk, min_size=32)
    blb_msk = asc_msk != spk_msk # for display
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     asc_nrm,
    #     gamma=0.5, opacity=0.5,
    #     )
    # vwr.add_image(
    #     asc_prd, colormap="inferno", visible=1,
    #     blending="additive", opacity=0.5,
    #     )
    # vwr.add_image(
    #     spk_msk, colormap="bop blue", visible=0,
    #     blending="additive", opacity=0.5,
    #     )
    # vwr.add_image(
    #     blb_msk, colormap="bop orange", visible=0,
    #     blending="additive", opacity=0.5,
    #     ) 
    
    # -------------------------------------------------------------------------
    
    from bdtools.norm import norm_pct
    from bdtools.nan import nan_replace
    from skimage.filters import gaussian
    from skimage.morphology import disk, binary_dilation
    
    rf = 1 / (2 ** (main_parameters["pvw_level"] - main_parameters["stk_level"]))
    asc_msk_rsc = rescale(asc_msk, rf)
    asc_nrm_rsc = rescale(asc_nrm, rf)
    asc_bkg_rsc = asc_nrm_rsc.copy()
    asc_bkg_rsc[binary_dilation(asc_msk_rsc, footprint=disk(3))] = np.nan
    asc_bkg_rsc = nan_replace(asc_bkg_rsc, kernel_size=3, parallel=False)
    asc_bkg_rsc = gaussian(asc_bkg_rsc, sigma=3)
    
    asc_bkg = rescale(asc_bkg_rsc, 1 / rf)
    asc_sub = asc_nrm - asc_bkg
    asc_sub = norm_pct(asc_sub, sample_fraction=0.1)
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     asc_nrm, visible=0,
    #     contrast_limits=[0, 1], gamma=0.5, opacity=1.0,
    #     )
    # vwr.add_image(
    #     asc_bkg, visible=0,
    #     contrast_limits=[0, 1], gamma=0.5, opacity=1.0,
    #     )
    # vwr.add_image(
    #     asc_sub, visible=1,
    #     gamma=0.5, opacity=1.0,
    #     )
    
    # -------------------------------------------------------------------------
    
    # Filter asc speckles
    spk_lbl = label(spk_msk)
    props = regionprops_table(
        spk_lbl, intensity_image=asc_sub,
        properties=("label", "centroid", "intensity_mean")
        )
    mask = props["intensity_mean"] > main_parameters["min_int"]
    valid_idx = props["label"][mask]
    coords = np.column_stack((
        np.round(props["centroid-0"][mask]).astype("int32"),
        np.round(props["centroid-1"][mask]).astype("int32"),
        ))
    valid_spk_msk = np.isin(spk_lbl, valid_idx)
    spk_msk[valid_spk_msk] = 0
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     asc_nrm, visible=0,
    #     contrast_limits=[0, 1], gamma=0.5, opacity=1.0,
    #     )
    # vwr.add_image(
    #     asc_sub, visible=1,
    #     gamma=0.5, opacity=1.0,
    #     )
    
    # vwr.add_image(
    #     blb_msk, colormap="blue", visible=1,
    #     blending="additive", opacity=0.5,
    #     ) 
    # vwr.add_image(
    #     spk_msk, colormap="bop blue", visible=1,
    #     blending="additive", opacity=0.5,
    #     )
    # vwr.add_image(
    #     valid_spk_msk, colormap="bop orange", visible=1,
    #     blending="additive", opacity=0.5,
    #     )
        
    # -------------------------------------------------------------------------
    
    # Get gaussian kernel
    kernel = get_gaussian_kernel(
        main_parameters["radius"], main_parameters["sigma"])
    # vwr = napari.Viewer()
    # vwr.add_image(kernel, colormap="plasma")
    
    # Validate speckles
    
    params = {
        "radius"    : 5, 
        "sigma"     : 1.0, 
        "min_corr"  : 0.8,
        "stk_level" : 1, 
        "pvw_level" : 4,
        }
    
    valid_coords = get_valid_speckles(asc_sub, pvw_msk, coords, **params)
    spk_dsp = get_speckle_display(
        asc_msk, coords, valid_coords, main_parameters["radius"])

    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        asc_nrm, visible=0,
        contrast_limits=[0, 1], gamma=0.5, opacity=1.0,
        )
    vwr.add_image(
        asc_sub, visible=1,
        gamma=0.5, opacity=1.0,
        )
    
    vwr.add_image(
        spk_dsp, colormap="bop orange", visible=1,
        blending="additive", opacity=1.0,
        )
    
    # vwr.add_image(
    #     asc_prd, colormap="inferno", visible=0,
    #     blending="additive", opacity=0.5,
    #     )
    # vwr.add_image(
    #     spk_msk, colormap="bop blue", visible=1,
    #     blending="additive", opacity=0.5,
    #     )
    # vwr.add_image(
    #     spk_dsp, colormap="bop blue", visible=1,
    #     blending="additive", opacity=1.0,
    #     )
    # vwr.add_image(
    #     blb_msk, colormap="bop orange", visible=1,
    #     blending="additive", opacity=0.5,
    #     ) 