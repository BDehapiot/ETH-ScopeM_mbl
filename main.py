#%% Imports -------------------------------------------------------------------

import gc
import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    imread_channel, norm_data, 
    get_slide_metadata, load_view, 
    get_preview_processed, get_preview_stats,
    get_obj_mask, get_obj_properties, 
    get_obj_crops, get_mask_from_coords,
    get_obj_intensities, get_asc_distances,
    format_ncl_data, format_asc_data,
    get_class_avg,
    plot,
    )

# bdtools
from bdtools.model.model import Model

# scipy
from scipy.stats import entropy

#%% Comments ------------------------------------------------------------------

"""
- The current work is to work on measure and analyse, refering to the previous
version of the code but also looking for simplifications. Still need to update
the plots, taking into account the addition of the speckle layer. BTW the asc
classification model is not enough annotated and would requires to process more
images. Also add plots to display.

- After classifying speckles I will need to filter out small asc that were not
recognized as speckles.
"""

#%% Inputs --------------------------------------------------------------------

main_parameters = {

    # Procedure
    "force_extract"         : 1,
    "force_predict_nuclei"  : 1,
    "force_predict_asc"     : 1,
    "force_classify_nuclei" : 1,
    "force_classify_asc"    : 1,
    "force_measure"         : 2,
    "force_analyse"         : 1,
    
    # Paths
    "data_path"             : Path("D:\local_mbl\data"),
    "model_pvw_name"        : "model-sm_preview_256_binary_227-1024",
    "model_ncl_name"        : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"        : "model-sm_asc_256_binary_307-1024",
    "model_ncl_class_name"  : "model-cls_nuclei_12_2_6_5045-8190b",
    "model_asc_class_name"  : "model-cls_asc_12_2_2_958-8192b",
    
    # extract()
    "pvw_level"             : 4,
    "stk_level"             : 1,
        
    # predict_nuclei()
    "ncl_level"             : 0,
    "ncl_thresh_0"          : 0.25,
    "ncl_thresh_1"          : 0.50,
    "ncl_min_size"          : 8,
    
    # predict_asc()
    "asc_level"             : 0,
    "asc_thresh_0"          : 0.25,
    "asc_thresh_1"          : None,
    "asc_min_size"          : 8,   
    
    # "min_int"               : 0.08, 
    # "radius"                : 5, 
    # "sigma"                 : 1.0, 
    # "min_corr"              : 0.8, 
    
    # Measure()
    "crop_size"             : 64,   # in pixels at stk_level = 0
    "norm_int"              : True, # normalize fluo. int. (per channel)
    
    # Analyse()
    "asc_bin_size"          : 8,    # in pixels at stk_level = 0
    "asc_bin_max"           : 64,   # in pixels at stk_level = 0
    
    # Maps  -------------------------------------------------------------------
    
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF",
        ],
    "ncl_classes" : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell",  
        ],
    "asc_classes" : [
        "Speckle", "All"
        ],
    
    "ncl_colors" : [
        [0.0, 0.0, 1.0], # Other
        [0.0, 0.8, 0.2], # Epithelium
        [1.0, 0.0, 0.0], # Neutrophil
        [1.0, 0.0, 1.0], # Macrophage
        [0.0, 1.0, 1.0], # Bcell
        [0.8, 0.8, 0.0], # Tcell
        ],

    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.extract()
        self.predict_nuclei()
        self.predict_asc()
        self.classify_nuclei()
        self.classify_asc()
        self.measure()
        self.analyse()
        
#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))

        self.qpt_paths = []
        self.mtd_paths = []
        self.vws_paths = []
        for p, qpt_path in enumerate(qpt_paths): 
            
            # Metadata --------------------------------------------------------
            
            # Paths
            mtd_name = str(qpt_path.name).replace(".qptiff", "_metadata.pkl")
            mtd_path = self.data_path / mtd_name
                        
            if not mtd_path.exists():
                mtd = get_slide_metadata(qpt_path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                    
            # Data ------------------------------------------------------------
                    
            for roi_idx in range(len(mtd["roi"])):
                
                # Paths
                vws_name = str(qpt_path.name).replace(".qptiff", f"_{roi_idx:02d}")
                vws_path = self.data_path / vws_name
                self.qpt_paths.append(qpt_path)
                self.mtd_paths.append(mtd_path)
                self.vws_paths.append(vws_path)
                
                pvw_raw_path = vws_path / "preview_raw.tif"
                pvw_msk_path = vws_path / "preview_mask.tif"
                pvw_nrm_path = vws_path / "preview_norm.tif"
                pvw_sts_path = vws_path / "preview_stats.pkl"
                stk_nrm_path = vws_path / "stack_norm.tif"

                if ((self.force_extract > 0 and not stk_nrm_path.exists()) 
                    or self.force_extract == 2):
                    
                    # Log
                    t0i = time.time()
                    log_str = f"Main() : extract() : {vws_path.name}"
                    print("-" * len(log_str))
                    print(log_str)
                    print("-" * len(log_str))
                    
                    # Model
                    if "model_preview" not in locals():
                        model_preview = Model(
                            parameters=None, 
                            model_path=Path.cwd() / self.model_pvw_name
                            )
                    
                    vws_path.mkdir(parents=True, exist_ok=True) 
                    
                    # Get previews --------------------------------------------
                    
                    t0 = time.time()
                    
                    # Preview raw
                    pvw_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.pvw_level, roi_idx=roi_idx,
                        )
                    
                    # Preview mask
                    pvw_prc = get_preview_processed(pvw_raw, channels=[2, 5, 6])
                    pvw_msk = model_preview.predict(pvw_prc, batch_size=32) > 0.5
                    pvw_msk = (pvw_msk * 255).astype("uint8")
                    
                    # Preview stats
                    pvw_sts = get_preview_stats(pvw_raw, pvw_msk, self.channels)
                    
                    # Preview norm.
                    pvw_nrm = (norm_data(pvw_raw, pvw_sts) * 255).astype("uint8")
                    
                    t1 = time.time()
                    print(f"{t1 - t0:.3f}s - get previews")
                    
                    # Load stack ----------------------------------------------
                    
                    t0 = time.time()
                    
                    stk_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.stk_level, roi_idx=roi_idx,
                        )
                    
                    t1 = time.time()
                    print(f"{t1 - t0:.3f}s - load stack")

                    # Norm stack ----------------------------------------------
                    
                    t0 = time.time()
                    
                    stk_nrm = (norm_data(stk_raw, pvw_sts) * 255).astype("uint8")
                    
                    t1 = time.time()
                    print(f"{t1 - t0:.3f}s - norm stack")
                          
                    # Save ----------------------------------------------------
                    
                    t0 = time.time()
                    
                    io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    io.imsave(stk_nrm_path, stk_nrm, check_contrast=False)
                    with open(pvw_sts_path, "wb")  as file:
                        pickle.dump(pvw_sts, file)     
                        
                    t1 = time.time()
                    print(f"{t1 - t0:.3f}s - save")
                        
                    t1f = time.time()
                    print(f"> {t1f - t0i:.3f}s")
                    
#%% Class(Main) : predict_nuclei() --------------------------------------------
                   
    def predict_nuclei(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            mtd_path = self.mtd_paths[p]
            qpt_path = self.qpt_paths[p]
            pvw_sts_path = vws_path / "preview_stats.pkl"
            stk_nrm_path = vws_path / "stack_norm.tif"
            ncl_msk_path = vws_path / "nuclei_mask.tif"
            ncl_prp_path = vws_path / "nuclei_properties.pkl"
            roi_idx = int(vws_path.stem[-2:])

            if ((self.force_predict_nuclei > 0 and not ncl_prp_path.exists()) 
                or self.force_predict_nuclei == 2):
                       
                # Log
                t0i = time.time()
                log_str = f"Main() : predict_nuclei() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_nuclei_path = Path.cwd() / self.model_ncl_name
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, model_path=model_nuclei_path)
                                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                ncl_nrm = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=self.ncl_level, roi_idx=roi_idx,
                    )
                ncl_nrm = norm_data(ncl_nrm, pvw_sts, channel="DAPI")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Predict -----------------------------------------------------
                
                t0 = time.time()

                ncl_prd = model_nuclei.predict(
                    ncl_nrm, patch_overlap=32, batch_size=32, chunk_size=5000)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - predict")
    
                # Get mask ----------------------------------------------------
    
                t0 = time.time()
    
                ncl_msk = get_obj_mask(
                    ncl_prd, 
                    thresh_0=self.ncl_thresh_0, 
                    thresh_1=self.ncl_thresh_1, 
                    min_size=self.ncl_min_size,
                    prd_level=self.ncl_level,
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get mask")
                
                # Get properties ----------------------------------------------
    
                t0 = time.time()
                
                ncl_nrm = imread_channel(stk_nrm_path, channel_idx=0)              
                ncl_prp = get_obj_properties(ncl_nrm, ncl_msk)

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get centroids")
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                # io.imsave(
                #     ncl_msk_path, 
                #     (ncl_msk * 255).astype("uint8"),
                #     check_contrast=False,
                #     )
                with open(ncl_prp_path, "wb") as file:
                    pickle.dump(ncl_prp, file)
    
                # Clear
                del ncl_nrm, ncl_prd, ncl_msk, ncl_prp
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : predict_asc() -----------------------------------------------
                   
    def predict_asc(self):
        
        for p, vws_path in enumerate(self.vws_paths): 
            
            # Paths
            mtd_path = self.mtd_paths[p]
            qpt_path = self.qpt_paths[p]
            pvw_sts_path = vws_path / "preview_stats.pkl"
            stk_nrm_path = vws_path / "stack_norm.tif"
            asc_msk_path = vws_path / "asc_mask.tif"
            asc_prp_path = vws_path / "asc_properties.pkl"
            roi_idx = int(vws_path.stem[-2:])
            
            if ((self.force_predict_asc > 0 and not asc_prp_path.exists()) 
                or self.force_predict_asc == 2):
                                
                # Log
                t0i = time.time()
                log_str = f"Main() : predict_asc() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_asc_path = Path.cwd() / self.model_asc_name
                if "model_asc" not in locals():
                    model_asc = Model(
                        parameters=None, model_path=model_asc_path)
                                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                asc_nrm = load_view(
                    qpt_path, metadata=mtd, channels="ASC", 
                    level=self.asc_level, roi_idx=roi_idx,
                    )
                asc_nrm = norm_data(asc_nrm, pvw_sts, channel="ASC")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Predict -----------------------------------------------------
                
                t0 = time.time()

                asc_prd = model_asc.predict(
                    asc_nrm, patch_overlap=32, batch_size=32, chunk_size=5000)
                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - predict")
                   
                # Get mask ----------------------------------------------------
    
                t0 = time.time()
    
                asc_msk = get_obj_mask(
                    asc_prd, 
                    thresh_0=self.asc_thresh_0, 
                    thresh_1=self.asc_thresh_1,
                    min_size=self.asc_min_size,
                    prd_level=self.asc_level,
                    )
                    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - process")  
                
                # Get properties ----------------------------------------------
    
                t0 = time.time()
                               
                asc_nrm = imread_channel(stk_nrm_path, channel_idx=4)    
                asc_prp = get_obj_properties(asc_nrm, asc_msk)

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get centroids")

                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                # io.imsave(
                #     asc_msk_path, 
                #     (asc_msk * 255).astype("uint8"),
                #     check_contrast=False,
                #     )
                with open(asc_prp_path, "wb") as file:
                    pickle.dump(asc_prp, file)
                
                # Clear
                del asc_nrm, asc_prd, asc_msk, asc_prp
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : classify_nuclei() -------------------------------------------

    def classify_nuclei(self):
        
        for p, vws_path in enumerate(self.vws_paths):
            
            # Paths
            stk_nrm_path = vws_path / "stack_norm.tif"
            ncl_prp_path = vws_path / "nuclei_properties.pkl"
            ncl_cls_path = vws_path / "nuclei_classes.pkl"
            
            if ((self.force_classify_nuclei > 0 and not ncl_cls_path.exists()) 
                or self.force_classify_nuclei == 2):
                
                # Log
                t0i = time.time()
                log_str = f"Main() : classify_nuclei() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_ncl_class_path = Path.cwd() / self.model_ncl_class_name
                if "model_ncl_class" not in locals():
                    model_ncl_class = Model(
                        parameters=None, model_path=model_ncl_class_path)

                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_prp_path, "rb") as file:
                    ncl_prp = pickle.load(file)
                stk_nrm = imread_channel(
                    stk_nrm_path, channel_idx=[1, 2, 3, 5, 6])  
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                ncl_crp = get_obj_crops(
                    stk_nrm, ncl_prp["y"], ncl_prp["x"],
                    crop_size=int(self.model_ncl_class_name.split("_")[2]),
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Classify ----------------------------------------------------
                
                t0 = time.time()
                
                ncl_cls = model_ncl_class.predict(ncl_crp)
                ncl_cls = {
                    "y"       : ncl_prp["y"], 
                    "x"       : ncl_prp["x"], 
                    "class"   : list(np.argmax(ncl_cls, axis=1)), 
                    "entropy" : list(entropy(
                        ncl_cls, axis=1, base=len(self.ncl_classes))), 
                    }
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - classify") 
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_cls_path, "wb") as file:
                    pickle.dump(ncl_cls, file)
                    
                # Clear
                del stk_nrm, ncl_prp, ncl_crp, ncl_cls
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : classify_asc() ----------------------------------------------

    def classify_asc(self):
        
        for p, vws_path in enumerate(self.vws_paths):
        
            # Paths
            stk_nrm_path = vws_path / "stack_norm.tif"
            asc_prp_path = vws_path / "asc_properties.pkl"
            asc_cls_path = vws_path / "asc_classes.pkl"
            
            if ((self.force_classify_asc > 0 and not asc_cls_path.exists()) 
                or self.force_classify_asc == 2):
                
                # Log
                t0i = time.time()
                log_str = f"Main() : classify_asc() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_asc_class_path = Path.cwd() / self.model_asc_class_name
                if "model_asc_class" not in locals():
                    model_asc_class = Model(
                        parameters=None, model_path=model_asc_class_path)
                    
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(asc_prp_path, "rb") as file:
                    asc_prp = pickle.load(file)
                asc_nrm = imread_channel(stk_nrm_path, channel_idx=4)  
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                asc_crp = get_obj_crops(
                    asc_nrm, asc_prp["y"], asc_prp["x"],
                    crop_size=int(self.model_asc_class_name.split("_")[2]),
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Classify ----------------------------------------------------
                
                t0 = time.time()
                
                asc_cls = model_asc_class.predict(asc_crp)
                asc_cls = {
                    "y"       : asc_prp["y"], 
                    "x"       : asc_prp["x"], 
                    "class"   : list(np.argmax(asc_cls, axis=1)), 
                    "entropy" : list(entropy(
                        asc_cls, axis=1, base=len(self.asc_classes))), 
                    }
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - classify") 
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(asc_cls_path, "wb") as file:
                    pickle.dump(asc_cls, file)
                    
                # Clear
                del asc_nrm, asc_prp, asc_crp, asc_cls
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : measure() ---------------------------------------------------

    def measure(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            stk_nrm_path  = vws_path / "stack_norm.tif"
            ncl_prp_path  = vws_path / "nuclei_properties.pkl"
            ncl_cls_path  = vws_path / "nuclei_classes.pkl" 
            ncl_int_path  = vws_path / "nuclei_intensities.pkl" 
            asc_prp_path  = vws_path / "asc_properties.pkl"  
            asc_cls_path  = vws_path / "asc_classes.pkl"  
            asc_int_path  = vws_path / "asc_intensities.pkl" 
            asc_dst_path  = vws_path / "asc_distances.pkl"
            
            ncl_data_path = vws_path / "nuclei_data.csv"
            ncl_data_avg_path = vws_path / "nuclei_data_avg.csv"
            ncl_data_std_path = vws_path / "nuclei_data_std.csv"
            ncl_data_sem_path = vws_path / "nuclei_data_sem.csv"
            asc_data_path = vws_path / "asc_data.csv"
            asc_data_avg_path = vws_path / "asc_data_avg.csv"
            asc_data_std_path = vws_path / "asc_data_std.csv"
            asc_data_sem_path = vws_path / "asc_data_sem.csv"
            
            if ((self.force_measure > 0 and not ncl_int_path.exists()) 
                or self.force_measure == 2):   
                
                # Log
                t0i = time.time()
                log_str = f"Main() : measure() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str)) 
                
                # Load --------------------------------------------------------
            
                t0 = time.time()

                stk_nrm = io.imread(stk_nrm_path).astype("float32") / 255
                with open(ncl_prp_path, "rb") as file: 
                    ncl_prp = pickle.load(file)
                with open(ncl_cls_path, "rb") as file: 
                    ncl_cls = pickle.load(file)
                with open(asc_prp_path, "rb") as file:
                    asc_prp = pickle.load(file)
                with open(asc_cls_path, "rb") as file:
                    asc_cls = pickle.load(file)
                    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
            
                # Get nuclei intensities --------------------------------------
                
                t0 = time.time()
                
                ncl_crp = get_obj_crops(
                    stk_nrm, ncl_prp["y"], ncl_prp["x"], 
                    crop_size=self.crop_size // 2 ** self.stk_level, 
                    )
                
                self.ncl_crp = ncl_crp
                
                ncl_int = {"y" : ncl_prp["y"], "x" : ncl_prp["x"]} 
                ncl_int.update(get_obj_intensities(
                    ncl_crp, self.channels, normalize=self.norm_int))

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei intensities")
                
                self.ncl_crp = ncl_crp
                
                # Get asc intensities -----------------------------------------
                    
                t0 = time.time()

                # Process ASC channel
                asc_msk = get_mask_from_coords(
                    asc_prp["coords"], shape=stk_nrm[0, ...].shape)
                asc_nrm = stk_nrm[4, ...] * asc_msk
                stk_nrm[4, ...] = asc_nrm
                
                # Measure int.
                asc_crp = get_obj_crops(
                    stk_nrm, asc_prp["y"], asc_prp["x"], 
                    crop_size=self.crop_size // 2 ** self.stk_level, 
                    )
                
                self.asc_crp = asc_crp
                
                asc_int = {"y" : asc_prp["y"], "x" : asc_prp["x"]} 
                asc_int.update(get_obj_intensities(
                    asc_crp, self.channels, normalize=self.norm_int))

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get asc intensities")
                                    
                # Get asc distances -------------------------------------------
                    
                t0 = time.time()
                
                asc_msk = get_mask_from_coords(
                    asc_prp["coords"], 
                    labels=np.array(asc_cls["class"]) + 1, 
                    shape=stk_nrm[0, ...].shape
                    )
                                   
                asc_dst = {
                    "y" : ncl_prp["y"],
                    "x" : ncl_prp["x"],
                    "asc_dst" : list(get_asc_distances(
                        asc_msk > 0 , ncl_prp, stk_level=1)),
                    "spk_dst" : list(get_asc_distances(
                        asc_msk == 1, ncl_prp, stk_level=1)),
                    }
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get asc distances")
                
                # Format data -------------------------------------------------
                
                t0 = time.time()
                
                ncl_data = format_ncl_data(
                    ncl_prp, ncl_cls, ncl_int, asc_dst, vws_path)
                asc_data = format_asc_data(
                    asc_prp, asc_cls, asc_int, vws_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - format data")
                
                # Class avg. data (per view) ----------------------------------
                
                t0 = time.time()
                
                ncl_data_avg, ncl_data_std, ncl_data_sem = get_class_avg(
                    ncl_data, self.ncl_classes, vws_path)
                asc_data_avg, asc_data_std, asc_data_sem = get_class_avg(
                    asc_data, self.asc_classes, vws_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - class avg. data (per view)")
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_int_path, "wb") as file:
                    pickle.dump(ncl_int, file)
                with open(asc_int_path, "wb") as file:
                    pickle.dump(asc_int, file)
                with open(asc_dst_path, "wb") as file:
                    pickle.dump(asc_dst, file)
                
                ncl_data.to_csv(ncl_data_path, index=False)
                ncl_data_avg.to_csv(ncl_data_avg_path, index=False)
                ncl_data_std.to_csv(ncl_data_std_path, index=False)
                ncl_data_sem.to_csv(ncl_data_sem_path, index=False)
                asc_data.to_csv(asc_data_path, index=False)
                asc_data_avg.to_csv(asc_data_avg_path, index=False)
                asc_data_std.to_csv(asc_data_std_path, index=False)
                asc_data_sem.to_csv(asc_data_sem_path, index=False)
                    
                # Clear
                del (
                    stk_nrm, 
                    ncl_prp, ncl_int,
                    asc_prp, asc_cls, asc_int, asc_dst,
                    ncl_data, ncl_data_avg, ncl_data_std, ncl_data_sem,
                    asc_data, asc_data_avg, asc_data_std, asc_data_sem,
                    )
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")

#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        if self.force_analyse > 0:
        
            # Log
            t0i = time.time()
            log_str = "Main() : analyse() :"
            print("-" * len(log_str))
            print(log_str)
            print("-" * len(log_str)) 
            
            # Paths
            ncl_data_avg_mrg_path = self.data_path / "nuclei_data_avg_merged.csv"
            ncl_data_avg_avg_path = self.data_path / "nuclei_data_avg_avg.csv"
            ncl_data_avg_std_path = self.data_path / "nuclei_data_avg_std.csv"
            ncl_data_avg_sem_path = self.data_path / "nuclei_data_avg_sem.csv"
            asc_data_avg_mrg_path = self.data_path / "asc_data_avg_merged.csv"
            asc_data_avg_avg_path = self.data_path / "asc_data_avg_avg.csv"
            asc_data_avg_std_path = self.data_path / "asc_data_avg_std.csv"
            asc_data_avg_sem_path = self.data_path / "asc_data_avg_sem.csv"
            
            # Class avg. data (merged) ----------------------------------------
            
            t0 = time.time()
            
            ncl_data_avg_mrg, asc_data_avg_mrg = [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_data_avg_path = vws_path / "nuclei_data_avg.csv"
                asc_data_avg_path = vws_path / "asc_data_avg.csv"
    
                # Load
                ncl_data_avg_mrg.append(pd.read_csv(ncl_data_avg_path))
                asc_data_avg_mrg.append(pd.read_csv(asc_data_avg_path))
                
            # Concatenate
            ncl_data_avg_mrg = pd.concat(ncl_data_avg_mrg, ignore_index=True)
            asc_data_avg_mrg = pd.concat(asc_data_avg_mrg, ignore_index=True)
                        
            # ///
            
            self.ncl_data_avg_mrg = ncl_data_avg_mrg
            self.asc_data_avg_mrg = asc_data_avg_mrg
            
            # ///
            
            # Class avg. data
            ncl_data_avg_avg, ncl_data_avg_std, ncl_data_avg_sem = get_class_avg(
                ncl_data_avg_mrg, self.ncl_classes, vws_path)
            asc_data_avg_avg, asc_data_avg_std, asc_data_avg_sem = get_class_avg(
                asc_data_avg_mrg, self.asc_classes, vws_path)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - class avg. data (merged)")
            
            # ///
            
            self.ncl_data_avg_avg = ncl_data_avg_avg
            self.ncl_data_avg_std = ncl_data_avg_std
            self.ncl_data_avg_sem = ncl_data_avg_sem
            self.asc_data_avg_avg = asc_data_avg_avg
            self.asc_data_avg_std = asc_data_avg_std
            self.asc_data_avg_sem = asc_data_avg_sem
            
            # ///
            
            # Save ------------------------------------------------------------
            
            t0 = time.time()
            
            ncl_data_avg_mrg.to_csv(ncl_data_avg_mrg_path, index=False)
            ncl_data_avg_avg.to_csv(ncl_data_avg_avg_path, index=False)
            ncl_data_avg_std.to_csv(ncl_data_avg_std_path, index=False)
            ncl_data_avg_sem.to_csv(ncl_data_avg_sem_path, index=False)
            
            asc_data_avg_mrg.to_csv(asc_data_avg_mrg_path, index=False)
            asc_data_avg_avg.to_csv(asc_data_avg_avg_path, index=False)
            asc_data_avg_std.to_csv(asc_data_avg_std_path, index=False)
            asc_data_avg_sem.to_csv(asc_data_avg_sem_path, index=False)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
        
#%% Class(Main) : plot() ------------------------------------------------------

    def plot(self):
        
        pass        
   
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)  
  
    # ncl_data = main.ncl_data
    # asc_data = main.asc_data
    # ncl_data_avg = main.ncl_data_avg
    # asc_data_avg = main.asc_data_avg
    ncl_data_avg_mrg = main.ncl_data_avg_mrg
    ncl_data_avg_avg = main.ncl_data_avg_avg
    ncl_data_avg_std = main.ncl_data_avg_std
    ncl_data_avg_sem = main.ncl_data_avg_sem
    asc_data_avg_mrg = main.asc_data_avg_mrg
    asc_data_avg_avg = main.asc_data_avg_avg
    asc_data_avg_std = main.asc_data_avg_std
    asc_data_avg_sem = main.asc_data_avg_sem
    
#%%

    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------

    ncl_all = ncl_data_avg_mrg
    ncl_avg = ncl_data_avg_avg
    ncl_sem = ncl_data_avg_sem
    asc_all = asc_data_avg_mrg
    asc_avg = asc_data_avg_avg
    asc_sem = asc_data_avg_sem

    # Inputs ------------------------------------------------------------------

    schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
    sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
    merged = ncl_all["view"].nunique(dropna=False) > 1

    hline_params = {
        "y" : 1, "color" : "0.75", "zorder" : 0,
        "linewidth" : 1, "linestyle" : ":",
        }
    
    grid_params = {
        "axis" : "x", "color" : "0.75", "zorder" : 0,
        "linewidth" : 1, "linestyle" : ":", 
        }

    # Nested function(s) ------------------------------------------------------
    
    def _ylim(ax, min_ymax=6):
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, np.maximum(min_ymax, ymax))
    
    def _barplot(ax, avg, x, color, err=None):
        ax.bar(x, avg, width=0.5, capsize=5, color=color, zorder=2)
        if not err is None:
            ax.errorbar(
                x, avg, yerr=np.vstack([err, err]),
                capsize=3, elinewidth=0.5, ecolor="0.5", fmt="none",
                )
        ax.axhline(**hline_params)
        ax.grid(**grid_params)
    
    def _boxplot(ax, vals, x, color, merged=False):
        ax.boxplot(
            vals, positions=[x], widths=0.5, patch_artist=True,
            boxprops=dict(color="0.5", facecolor="none", linewidth=1),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color="0.5", linewidth=1),
            capprops=dict(color="0.5", linewidth=1),
            showfliers=False,
            )
        ax.axhline(**hline_params)
        ax.grid(**grid_params)    
    
    # Execute -----------------------------------------------------------------
    
    # plt.ioff()
    
    # Initialize plot
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 12))
    axes = axes.flatten()
    
    # Main title/info ---------------------------------------------------------
       
    # Main title    
    fig.suptitle(
        "Merged views" if merged else ncl_all["view"].iloc[0],
        fontsize=16, fontweight="bold", color="k",
        y=0.92, x=0.025, ha="left",
        )

    # Secondary text
    text = (
        f"n views = {ncl_all['view'].nunique()} | "
        f"n nuclei = {ncl_all['count'].sum():,} | "
        f"avg. entropy = {ncl_all['entropy'].mean():.3f}"
        )
    fig.text(
        0.025, 0.89, text,
        fontsize=12, color="0.5",
        ha="left", va="top",
        )
    
    # (0) Class predictions frequency -----------------------------------------
    
    ax = axes[0]

    for cl, cla in enumerate(main.ncl_classes):
        _barplot(
            ax, ncl_avg["freq"], sclasses, 
            main.ncl_colors, err=ncl_sem["freq"]
            )
        
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class frequency", pad=10)
    
    # (1) Class predictions entropy -------------------------------------------
    
    ax = axes[1]
    
    for cl, cla in enumerate(main.ncl_classes):
        vals = np.array(ncl_all[ncl_all["class"] == cla]["entropy"])
        _boxplot(ax, vals, cl, main.ncl_colors[cl])
        
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Entropy")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class pred. entropy", pad=10)
    
    # (2) ??? -----------------------------------------------------------------
    
    ax = axes[2]
    ax.axis("off")
    ax.set_facecolor("none")
    
    # (3 - 8) Channel intensities ---------------------------------------------
    
    for cl, cla in enumerate(main.ncl_classes):

        ax = axes[cl + 3]                   
        
        for ch, chn in enumerate(schannels):
            vals = np.array(ncl_all[ncl_all["class"] == cla][f"{chn}_int"])
            _boxplot(ax, vals, ch, main.ncl_colors[cl])
                    
        # Legend
        _ylim(ax)
        ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
        ax.set_xticklabels(schannels, rotation=45, ha="center")
        ax.set_title(cla, pad=10)
    
    # (9) Class ASC intensities -----------------------------------------------
    
    ax = axes[9]
        
    for cl, cla in enumerate(main.ncl_classes):
        _barplot(
            ax, ncl_avg["ASC_int"], sclasses, 
            main.ncl_colors, err=ncl_sem["ASC_int"]
            )
        
    # Legend
    _ylim(ax)
    ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class ASC int.", pad=10)
    
    # -------------------------------------------------------------------------
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.close(fig) 
    # plt.ion()
    
#%%
    
    # def get_class_avg(data, classes, vws_path):  
    
    #     def _get_class_avg(grp, method="mean"):
            
    #         # Statistics
    #         if method == "mean": df = grp.mean(numeric_only=True)
    #         if method == "std" : df = grp.std(numeric_only=True)
    #         if method == "sem" : df = grp.sem(numeric_only=True)
                
    #         # Count & frequency
    #         if "y" in data.columns:
    #             count = grp.size()
    #             freq = count / len(data)
    #         else:
    #             count = grp["count"].sum()
    #             freq = df["freq"]
    
    #         # Add count & frequency
    #         df = df.assign(count=count)
    #         df = df.assign(freq=freq)
    
    #         # Format
    #         if "y" in data.columns:
    #             df.index = df.index.map(lambda i: classes[i])
    #             df = df.reset_index(names="class").assign(view=vws_path.name)
    #         else:
    #             df = df.reset_index(names="class")
    
    #         cols = [c for c in ["view", "class", "count", "freq"] if c in df.columns]
    #         features = [c for c in df.columns if c not in cols]
            
    #         return df[cols + features]
    
    #     # Group per class
    #     grp = data.drop(columns=["x", "y"], errors="ignore")
    #     grp = grp.groupby("class", sort=False)
            
    #     # Format
    #     avg = _get_class_avg(grp, method="mean")
    #     std = _get_class_avg(grp, method="std")
    #     sem = _get_class_avg(grp, method="sem")
    
    #     return avg, std, sem
    
    # Paths
    
    # Load
    vws_path = main.vws_paths[0]
    stk_nrm_path = vws_path / "stack_norm.tif"
    asc_prp_path = vws_path / "asc_properties.pkl"
    ncl_data_path = vws_path / "nuclei_data.csv"
    ncl_data_avg_path = vws_path / "nuclei_data_avg.csv"
    
    stk_nrm = io.imread(stk_nrm_path)
    with open(asc_prp_path, "rb") as file: 
        asc_prp = pickle.load(file)
    ncl_data = pd.read_csv(ncl_data_path)
    ncl_data_avg = pd.read_csv(ncl_data_avg_path)
    
    asc_msk = get_mask_from_coords(
        asc_prp["coords"], shape=stk_nrm[0, ...].shape)
    asc_nrm = stk_nrm[4, ...] * asc_msk
    
    import napari
    vwr = napari.Viewer()
    vwr.add_image(asc_nrm)
    
    # # Run
    # avg, std, sem = get_class_avg(
    #     ncl_data, main.ncl_classes, vws_path)
    # avg_mrg, std_mrg, sem_mrg = get_class_avg(
    #     ncl_data_avg_mrg, main.ncl_classes, vws_path)

