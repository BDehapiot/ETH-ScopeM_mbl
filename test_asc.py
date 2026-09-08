#%% Imports -------------------------------------------------------------------

import time
import pickle
from skimage import io
from pathlib import Path
from main import main_parameters

# functions
from functions import load_view, norm_data, get_asc_mask

# bdtools
from bdtools.model.model import Model

# skimage
from skimage.transform import rescale

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

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
    asc_msk_path = vws_path / "asc_mask.tif"
    asc_prc_path = vws_path / "asc_processed.tif"
    asc_dsp_path = vws_path / "asc_display.tif"
    roi_idx = int(vws_path.stem[-2:])

    # Model
    model_asc_path = Path.cwd() / model_asc_name
    if "model_asc" not in locals():
        model_asc = Model(
            parameters=None, model_path=model_asc_path)
                    
    # Load --------------------------------------------------------------------
    
    with open(mtd_path, "rb") as file:
        mtd = pickle.load(file)
    with open(pvw_sts_path, "rb") as file:
        pvw_sts = pickle.load(file)
    pvw_msk = io.imread(pvw_msk_path)
    asc_raw = load_view(
        qpt_path, metadata=mtd, channels="ASC", 
        level=main_parameters["asc_level"], roi_idx=roi_idx,
        )
    
    asc_nrm = norm_data(asc_raw, pvw_sts, channel="ASC")
    
    # Predict -----------------------------------------------------------------
    
    asc_prd = model_asc.predict(
        asc_nrm, patch_overlap=32, batch_size=32, chunk_size=5000)
                                       
#%%

    params = {
        "asc_level"         : 0,
        "stk_level"         : 1,
        "pvw_level"         : 4,
        "asc_thresh"        : 0.05,
        "min_dist"          : 5,   
        "min_int"           : 0.08, 
        "radius"            : 5, 
        "sigma"             : 1.0, 
        "min_corr"          : 0.8, 
        }

    # get_asc_msk()
    t0 = time.time()
    print("get_asc_msk() : ", end="", flush=False)
    asc_msk, asc_prc, asc_dsp = get_asc_mask(
        asc_prd, asc_nrm, pvw_msk, **params
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # rescale asc_nrm
    rf = 1 / (2 ** (params["stk_level"] - params["asc_level"]))
    asc_nrm = rescale(asc_nrm, rf)
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(asc_nrm)
    vwr.add_image(asc_dsp)
    
    
    