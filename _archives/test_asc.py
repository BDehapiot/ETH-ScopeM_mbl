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
    "asc_thresh" : 0.01,
    "min_dist"   : 5,   
    "min_int"    : 0.075, 
    "radius"     : 5, 
    "sigma"      : 1.0, 
    "min_corr"   : 0.8, 
    
    }

main_parameters.update(test_parameters)

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
    
    t1f = time.time()
    print(f"> {t1f - t0i:.3f}s")
    
#%% Function : get_asc_mask() -------------------------------------------------

def get_asc_mask(
        asc_nrm, asc_prd, pvw_msk, asc_thresh=0.05,
        pvw_level=4, stk_level=1, asc_level=0,
        ):
    
    # Rescale
    rf = 1 / (2 ** (stk_level - asc_level))
    asc_prd = rescale(asc_prd, rf)
    rf = 1 / (2 ** (stk_level - pvw_level))
    pvw_msk = rescale(pvw_msk, rf, order=0)
    
    # Get mask
    asc_msk = asc_prd > asc_thresh
    asc_prc = asc_nrm.copy()
    asc_prc[asc_msk == 0] = 0
    
    # Separate speckles from blobs
    asc_msk = remove_small_objects(asc_msk, min_size=4)
    spk_msk = asc_msk != remove_small_objects(asc_msk, min_size=32)
    
    # Update asc mask
    asc_msk = asc_msk.astype("uint8")
    asc_msk[spk_msk] = 2
    asc_msk[pvw_msk == 0] = 0
    
    return asc_msk, asc_prc
    
#%% Execute -------------------------------------------------------------------
    
if __name__ == "__main__":

    asc_thresh = main_parameters["asc_thresh"]
    pvw_level = main_parameters["pvw_level"]
    stk_level = main_parameters["stk_level"]   
    asc_level = main_parameters["asc_level"]    

    t0 = time.time()
    
    asc_nrm = stk_nrm[4, ...].astype("float32") / 255
    asc_msk, asc_prc = get_asc_mask(
        asc_nrm, asc_prd, pvw_msk, asc_thresh=asc_thresh,
        pvw_level=pvw_level, stk_level=stk_level, asc_level=asc_level,
        )
    asc_msk = asc_msk * 100
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s - get mask")
       
    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        asc_nrm,
        gamma=0.5, opacity=1.0,
        )
    vwr.add_image(
        asc_msk, colormap="bop orange", visible=1,
        blending="additive", opacity=0.5,
        )
    