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
    imread_channel, norm_data, get_sfactor,
    get_slide_metadata, load_view, 
    get_preview_processed, get_preview_stats,
    get_obj_mask, get_obj_properties, 
    get_obj_crops, get_mask_from_coords,
    get_ncl_intensities, get_ncl_distances, get_asc_stats, 
    format_ncl_data, format_asc_data, average_data,
    bin_ncl_distances, average_ncl_distances_bin,
    )

# bdtools
from bdtools.model.model import Model

# scipy
from scipy.stats import entropy

#%% Comments ------------------------------------------------------------------

"""
distances & areas in .pkl files are in level = stk_level 
distances & areas in .csv files are in level = 0
"""

#%% Inputs --------------------------------------------------------------------

main_parameters = {

    # Procedure
    "force_extract"         : 0,
    "force_predict_nuclei"  : 0,
    "force_predict_asc"     : 0,
    "force_classify_nuclei" : 0,
    "force_classify_asc"    : 0,
    "force_measure"         : 0,
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
    
    # Measure()
    "pixel_size"            : 0.5,  # in µm at level = 0
    "int_crop_size"         : 32,   # in µm at level = 0
    "int_norm"              : True, # normalize fluo. int. (per channel)
    
    # Analyse()
    "dst_bin_size"          : 4,    # in µm at level = 0
    "dst_bin_max"           : 32,   # in µm at level = 0
    "dst_norm"              : True, # normalize distances
    
    # Maps  -------------------------------------------------------------------
    
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF",
        ],
    "ncl_classes" : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell",  
        ],
    "asc_classes" : [
        "Speckle", "Non-speckle"
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
                    "class"   : np.argmax(ncl_cls, axis=1), 
                    "entropy" : entropy(
                        ncl_cls, axis=1, base=len(self.ncl_classes)), 
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
                    "class"   : np.argmax(asc_cls, axis=1), 
                    "entropy" : entropy(
                        asc_cls, axis=1, base=len(self.asc_classes)), 
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

            pvw_msk_path = vws_path / "preview_mask.tif"
            stk_nrm_path = vws_path / "stack_norm.tif"
            ncl_prp_path = vws_path / "nuclei_properties.pkl"
            ncl_cls_path = vws_path / "nuclei_classes.pkl" 
            asc_prp_path = vws_path / "asc_properties.pkl"  
            asc_cls_path = vws_path / "asc_classes.pkl"  

            ncl_int_path = vws_path / "nuclei_intensities.pkl" 
            ncl_dst_path = vws_path / "nuclei_distances.pkl"
            
            asc_sts_path = vws_path / "asc_stats.csv"

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

                pvw_msk = io.imread(pvw_msk_path).astype(bool)
                stk_nrm = io.imread(stk_nrm_path).astype("float32") / 255
                with open(ncl_prp_path, "rb") as file: 
                    ncl_prp = pickle.load(file)
                with open(ncl_cls_path, "rb") as file: 
                    ncl_cls = pickle.load(file)
                with open(asc_prp_path, "rb") as file:
                    asc_prp = pickle.load(file)
                with open(asc_cls_path, "rb") as file:
                    asc_cls = pickle.load(file)
                    
                # Get asc_msk
                asc_msk = get_mask_from_coords(
                    asc_prp["coords"], 
                    labels=asc_cls["class"] + 1,
                    shape=stk_nrm[0, ...].shape
                    )
                    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                                
                # Get nuclei intensities --------------------------------------
                
                t0 = time.time()
                
                ncl_int = get_ncl_intensities(
                    stk_nrm, ncl_prp, asc_prp, asc_msk, self.channels, 
                    crop_size=self.int_crop_size, normalize=self.int_norm,
                    )

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei intensities")
                                                        
                # Get nuclei distances (to asc) -------------------------------
                    
                t0 = time.time()
                                   
                ncl_dst = get_ncl_distances(
                    asc_msk, ncl_prp, stk_level=self.stk_level)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei distances")
                                    
                # Get asc statistics ------------------------------------------
                
                t0 = time.time()
                
                asc_sts = get_asc_stats(
                    pvw_msk, ncl_prp, asc_prp, asc_cls, vws_path, 
                    pvw_level=self.pvw_level, pixel_size=self.pixel_size,
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get asc statistics")
                
                # Format/avg. data --------------------------------------------
                
                t0 = time.time()
                
                # ncl_data
                ncl_data = format_ncl_data(
                    ncl_prp, ncl_cls, ncl_int, ncl_dst, vws_path)
                ncl_data_avg, ncl_data_std, ncl_data_sem = average_data(
                    ncl_data, self.ncl_classes, vws_path)
                
                # asc_data
                asc_data = format_asc_data(
                    asc_prp, asc_cls, vws_path)
                asc_data_avg, asc_data_std, asc_data_sem = average_data(
                    asc_data, self.asc_classes, vws_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - format/avg. data")
                
                # Save --------------------------------------------------------
                                
                t0 = time.time()
                
                with open(ncl_int_path, "wb") as file:
                    pickle.dump(ncl_int, file)
                with open(ncl_dst_path, "wb") as file:
                    pickle.dump(ncl_dst, file)
                
                asc_sts.to_csv(asc_sts_path, index=False)
                
                ncl_data.to_csv(ncl_data_path, index=False)
                ncl_data_avg.to_csv(ncl_data_avg_path, index=False)
                ncl_data_std.to_csv(ncl_data_std_path, index=False)
                ncl_data_sem.to_csv(ncl_data_sem_path, index=False)
                
                asc_data.to_csv(asc_data_path, index=False)
                asc_data_avg.to_csv(asc_data_avg_path, index=False)
                asc_data_std.to_csv(asc_data_std_path, index=False)
                asc_data_sem.to_csv(asc_data_sem_path, index=False)
                    
                # ///
                
                self.vws_path = vws_path 
                self.stk_nrm  = stk_nrm
                self.ncl_prp  = ncl_prp
                self.ncl_cls  = ncl_cls
                self.asc_prp  = asc_prp
                self.asc_cls  = asc_cls
                
                self.ncl_int  = ncl_int
                self.ncl_dst  = ncl_dst
                
                self.asc_sts  = asc_sts
                
                self.ncl_data = ncl_data
                self.ncl_data_avg  = ncl_data_avg
                self.ncl_data_std  = ncl_data_std
                self.ncl_data_sem  = ncl_data_sem
                
                self.asc_data = asc_data
                self.asc_data_avg  = asc_data_avg
                self.asc_data_std  = asc_data_std
                self.asc_data_sem  = asc_data_sem
                
                # ///
                
                # Clear
                del (
                    stk_nrm, ncl_prp, ncl_cls, asc_prp, asc_cls,
                    ncl_int, ncl_dst,
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
            asc_sts_mrg_path = self.data_path / "asc_stats_merged.csv"
            
            ncl_dst_bin_avg_path = self.data_path / "nuclei_distances_bin_avg.csv"
            ncl_dst_bin_std_path = self.data_path / "nuclei_distances_bin_std.csv"
            ncl_dst_bin_sem_path = self.data_path / "nuclei_distances_bin_sem.csv"
            
            # Load & format/avg. data -----------------------------------------
            
            t0 = time.time()
            
            ncl_data_avg_mrg, asc_data_avg_mrg, asc_sts_mrg = [], [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_data_avg_path = vws_path / "nuclei_data_avg.csv"
                asc_data_avg_path = vws_path / "asc_data_avg.csv"
                asc_sts_path = vws_path / "asc_stats.csv" 
    
                # Load
                ncl_data_avg_mrg.append(pd.read_csv(ncl_data_avg_path))
                asc_data_avg_mrg.append(pd.read_csv(asc_data_avg_path))
                asc_sts_mrg.append(pd.read_csv(asc_sts_path))
                
            # Concatenate
            ncl_data_avg_mrg = pd.concat(ncl_data_avg_mrg, ignore_index=True)
            asc_data_avg_mrg = pd.concat(asc_data_avg_mrg, ignore_index=True)
            asc_sts_mrg = pd.concat(asc_sts_mrg, ignore_index=True)
            
            # ///
            
            self.asc_sts_mrg = asc_sts_mrg
            
            # ///
            
            # Class avg. data
            ncl_data_avg_avg, ncl_data_avg_std, ncl_data_avg_sem = average_data(
                ncl_data_avg_mrg, self.ncl_classes, vws_path)
            asc_data_avg_avg, asc_data_avg_std, asc_data_avg_sem = average_data(
                asc_data_avg_mrg, self.asc_classes, vws_path)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - load & format/avg. data")

            # Bin ncl distances -----------------------------------------------
            
            t0 = time.time()
            
            ncl_dst_bin_mrg = []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_cls_path = vws_path / "nuclei_classes.pkl"
                ncl_dst_path = vws_path / "nuclei_distances.pkl"
                
                # Load
                with open(ncl_cls_path, "rb") as file: 
                    ncl_cls = pickle.load(file)
                with open(ncl_dst_path, "rb") as file: 
                    ncl_dst = pickle.load(file)
                    
                # Bin ncl distances
                ncl_dst_bin = bin_ncl_distances(
                    ncl_dst, ncl_cls, self.ncl_classes,
                    bin_size=self.dst_bin_size, 
                    bin_max=self.dst_bin_max, 
                    normalize=self.dst_norm, 
                    ) 
                                
                ncl_dst_bin_mrg.append(ncl_dst_bin)
                
                # ///
                
                self.ncl_cls = ncl_cls
                self.ncl_dst = ncl_dst
                self.ncl_dst_bin = ncl_dst_bin
                self.ncl_dst_bin_mrg = ncl_dst_bin_mrg
                
                # ///
                
            ncl_dst_bin_avg, ncl_dst_bin_std, ncl_dst_bin_sem = (
                average_ncl_distances_bin(ncl_dst_bin_mrg))
            
            # ///
            
            self.ncl_dst_bin_avg = ncl_dst_bin_avg
            self.ncl_dst_bin_std = ncl_dst_bin_std
            self.ncl_dst_bin_sem = ncl_dst_bin_sem
            
            # ///     
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - bin ncl distances")
            
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
            asc_sts_mrg.to_csv(asc_sts_mrg_path, index=False)
            
            ncl_dst_bin_avg.to_csv(ncl_dst_bin_avg_path)
            ncl_dst_bin_std.to_csv(ncl_dst_bin_std_path)
            ncl_dst_bin_sem.to_csv(ncl_dst_bin_sem_path)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
            
            # ///
                        
            self.ncl_data_avg_mrg = ncl_data_avg_mrg
            self.ncl_data_avg_avg = ncl_data_avg_avg
            self.ncl_data_avg_std = ncl_data_avg_std
            self.ncl_data_avg_sem = ncl_data_avg_sem
            self.asc_data_avg_mrg = asc_data_avg_mrg
            self.asc_data_avg_avg = asc_data_avg_avg
            self.asc_data_avg_std = asc_data_avg_std
            self.asc_data_avg_sem = asc_data_avg_sem
            
            # ///
        
#%% Class(Main) : plot() ------------------------------------------------------

    def plot(self):
        
        pass        
   
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)  
    
    # vws_path = main.vws_path
    # pvw_msk  = main.pvw_msk
    # stk_nrm  = main.stk_nrm
    
    # ncl_prp  = main.ncl_prp
    # ncl_cls  = main.ncl_cls
    # asc_prp  = main.asc_prp
    # asc_cls  = main.asc_cls
    # ncl_int  = main.ncl_int
    # ncl_dst  = main.ncl_dst
    # asc_sts  = main.asc_sts
    # ncl_data = main.ncl_data
    # ncl_data_avg = main.ncl_data_avg
    # ncl_data_std = main.ncl_data_std
    # ncl_data_sem = main.ncl_data_sem
    # asc_data = main.asc_data
    # asc_data_avg = main.asc_data_avg
    # asc_data_std = main.asc_data_std
    # asc_data_sem = main.asc_data_sem

    asc_sts_mrg = main.asc_sts_mrg
    # ncl_data_avg_mrg = main.ncl_data_avg_mrg
    # ncl_data_avg_avg = main.ncl_data_avg_avg
    # ncl_data_avg_std = main.ncl_data_avg_std
    # ncl_data_avg_sem = main.ncl_data_avg_sem
    # asc_data_avg_mrg = main.asc_data_avg_mrg
    # asc_data_avg_avg = main.asc_data_avg_avg
    # asc_data_avg_std = main.asc_data_avg_std
    # asc_data_avg_sem = main.asc_data_avg_sem
    
    # ncl_dst_bin = main.ncl_dst_bin
    # ncl_dst_bin_mrg = main.ncl_dst_bin_mrg
    # ncl_dst_bin_avg = main.ncl_dst_bin_avg
    # ncl_dst_bin_std = main.ncl_dst_bin_std
    # ncl_dst_bin_sem = main.ncl_dst_bin_sem
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(pvw_msk)
            
#%%

    # def get_asc_stats(
    #     pvw_msk, ncl_prp, asc_prp, asc_cls,
    #     vws_path, pvw_level=4, stk_level=1, 
    #     ): 

    #     # Init.
    #     asc_sts = {}
    #     pvw_factor = (2 ** pvw_level) ** 2
    #     asc_factor = (2 ** stk_level) ** 2
    #     spk_idx = np.where(asc_cls["class"] == 0)[0]
            
    #     # Areas
    #     all_area = np.sum(pvw_msk > 0) * pvw_factor
    #     asc_area = np.sum(asc_prp["area"]) * asc_factor
    #     asc_area_ratio = asc_area / all_area
    #     spk_area = np.sum(asc_prp["area"][spk_idx]) * asc_factor
    #     spk_area_ratio = spk_area / all_area    
        
    #     # Counts
    #     ncl_num = len(ncl_prp["y"])
    #     asc_num = len(asc_prp["y"])
    #     spk_num = (asc_cls["class"] == 0).sum()
    #     asc_num_ratio = asc_num / ncl_num
    #     spk_num_ratio = spk_num / ncl_num
                
    #     # Append
    #     asc_sts["all_area"      ] = all_area
    #     asc_sts["asc_area"      ] = asc_area
    #     asc_sts["asc_area_ratio"] = asc_area_ratio
    #     asc_sts["spk_area_ratio"] = spk_area_ratio
    #     asc_sts["ncl_num"       ] = ncl_num
    #     asc_sts["asc_num"       ] = asc_num
    #     asc_sts["spk_num"       ] = spk_num
    #     asc_sts["asc_num_ratio" ] = asc_num_ratio
    #     asc_sts["spk_num_ratio" ] = spk_num_ratio
        
    #     # Format
    #     asc_sts = pd.DataFrame([asc_sts])
    #     asc_sts.insert(0, "view", vws_path.name)
        
    #     return asc_sts
    
    # -------------------------------------------------------------------------
    
    def get_asc_stats(
        pvw_msk, ncl_prp, asc_prp, asc_cls, vws_path,
        pvw_level=4, pixel_size=0.5,
        ): 

        # Init.
        asc_sts = {}
        spk_idx = np.where(asc_cls["class"] == 0)[0]
        nsp_idx = np.where(asc_cls["class"] == 1)[0]
        sfactor = get_sfactor(pvw_level, 0, ndim=2, pixel_size=pixel_size)
            
        # Areas
        tissue_area = np.sum(pvw_msk > 0) * sfactor
        asc_all_area = np.sum(asc_prp["area"])
        asc_spk_area = np.sum(asc_prp["area"][spk_idx])
        asc_nsp_area = np.sum(asc_prp["area"][nsp_idx])
        asc_all_area_ratio = asc_all_area / tissue_area
        asc_spk_area_ratio = asc_spk_area / tissue_area  
        asc_nsp_area_ratio = asc_nsp_area / tissue_area  
        
        # Counts
        ncl_num = len(ncl_prp["y"])
        asc_all_num = len(asc_prp["y"])        
        asc_spk_num = (asc_cls["class"] == 0).sum()
        asc_nsp_num = (asc_cls["class"] == 1).sum()
        asc_all_num_ratio = asc_all_num / ncl_num
        asc_spk_num_ratio = asc_spk_num / ncl_num
        asc_nsp_num_ratio = asc_nsp_num / ncl_num
  
        # Append
        asc_sts["tissue_area"       ] = tissue_area
        asc_sts["asc_all_area"      ] = asc_all_area
        asc_sts["asc_all_area_ratio"] = asc_all_area_ratio
        asc_sts["asc_spk_area"      ] = asc_spk_area
        asc_sts["asc_spk_area_ratio"] = asc_spk_area_ratio
        asc_sts["asc_nsp_area"      ] = asc_nsp_area
        asc_sts["asc_nsp_area_ratio"] = asc_nsp_area_ratio
        
        asc_sts["ncl_num"           ] = ncl_num
        asc_sts["asc_all_num"       ] = asc_all_num
        asc_sts["asc_spk_num"       ] = asc_spk_num
        asc_sts["asc_nsp_num"       ] = asc_nsp_num
        asc_sts["asc_all_num_ratio" ] = asc_all_num_ratio
        asc_sts["asc_spk_num_ratio" ] = asc_spk_num_ratio
        asc_sts["asc_nsp_num_ratio" ] = asc_nsp_num_ratio
        
        # Format
        asc_sts = pd.DataFrame([asc_sts])
        asc_sts.insert(0, "view", vws_path.name)
        
        return asc_sts
    
    # -------------------------------------------------------------------------
    
    # Paths
    vws_path = main.vws_paths[0]
    pvw_msk_path = vws_path / "preview_mask.tif"
    ncl_prp_path = vws_path / "nuclei_properties.pkl"
    ncl_cls_path = vws_path / "nuclei_classes.pkl"
    ncl_data_path = vws_path / "nuclei_data.csv"
    asc_prp_path = vws_path / "asc_properties.pkl"
    asc_cls_path = vws_path / "asc_classes.pkl"
    asc_data_path = vws_path / "asc_data.csv"
    
    
    # Load
    pvw_msk = io.imread(pvw_msk_path)
    with open(ncl_prp_path, "rb") as file: 
        ncl_prp = pickle.load(file)
    with open(ncl_cls_path, "rb") as file: 
        ncl_cls = pickle.load(file)
    with open(asc_prp_path, "rb") as file: 
        asc_prp = pickle.load(file)
    with open(asc_cls_path, "rb") as file: 
        asc_cls = pickle.load(file)
    ncl_data = pd.read_csv(ncl_data_path)
    asc_data = pd.read_csv(asc_data_path)
    
    # Run
    asc_sts = get_asc_stats(
        pvw_msk, ncl_prp, asc_prp, asc_cls,
        vws_path, pvw_level=4, pixel_size=0.5,
        )   
    
#%%

    # import matplotlib.pyplot as plt

    # # -------------------------------------------------------------------------

    # ncl_all = ncl_data_avg_mrg
    # ncl_avg = ncl_data_avg_avg
    # ncl_sem = ncl_data_avg_sem
    # asc_all = asc_data_avg_mrg
    # asc_avg = asc_data_avg_avg
    # asc_sem = asc_data_avg_sem
    # dst_avg = ncl_dst_bin_avg
    # dst_sem = ncl_dst_bin_sem

    # # Inputs ------------------------------------------------------------------

    # schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
    # sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
    # merged = ncl_all["view"].nunique(dropna=False) > 1

    # hline_params = {
    #     "y" : 1, "color" : "0.75", "zorder" : 0,
    #     "linewidth" : 1, "linestyle" : ":",
    #     }
    
    # grid_params = {
    #     "axis" : "x", "color" : "0.75", "zorder" : 0,
    #     "linewidth" : 1, "linestyle" : ":", 
    #     }

    # # Nested function(s) ------------------------------------------------------
    
    # def _ylim(ax, min_ymax=6):
    #     _, ymax = ax.get_ylim()
    #     ax.set_ylim(0, np.maximum(min_ymax, ymax))
    
    # def _barplot(
    #         ax, avg, x, err=None,
    #         width=0.5, shift=0.0, color="gray"
    #         ):
        
    #     x_pos = np.arange(len(x)) + shift
    #     ax.bar(
    #         x_pos, avg, width=width,
    #         color=color, capsize=5, zorder=2,
    #         )
        
    #     if not err is None:
    #         ax.errorbar(
    #             x_pos, avg, yerr=np.vstack([err, err]),
    #             capsize=3, elinewidth=0.5, ecolor="0.5", fmt="none",
    #             )
            
    #     ax.axhline(**hline_params)
    #     ax.grid(**grid_params)
    
    # def _boxplot(ax, vals, x, merged=False, color="gray"):
        
    #     ax.boxplot(
    #         vals, positions=[x], widths=0.5, patch_artist=True,
    #         boxprops=dict(color="0.5", facecolor="none", linewidth=1),
    #         medianprops=dict(color=color, linewidth=2),
    #         whiskerprops=dict(color="0.5", linewidth=1),
    #         capprops=dict(color="0.5", linewidth=1),
    #         showfliers=False,
    #         )
        
    #     ax.axhline(**hline_params)
    #     ax.grid(**grid_params)    
    
    # def _lineplot(ax, vals, x, color, label):
    #     ax.plot(x, vals, color=color, label=label)
    #     ax.grid(**grid_params)
    
    # # Execute -----------------------------------------------------------------
    
    # # plt.ioff()
    
    # # Initialize plot
    # fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 12))
    # axes = axes.flatten()
    
    # # Main title/info ---------------------------------------------------------
       
    # # Main title    
    # fig.suptitle(
    #     "Merged views" if merged else ncl_all["view"].iloc[0],
    #     fontsize=16, fontweight="bold", color="k",
    #     y=0.92, x=0.025, ha="left",
    #     )

    # # Secondary text
    # text = (
    #     f"n views = {ncl_all['view'].nunique()} | "
    #     f"n nuclei = {ncl_all['count'].sum():,} | "
    #     f"avg. entropy = {ncl_all['entropy'].mean():.3f}"
    #     )
    # fig.text(
    #     0.025, 0.89, text,
    #     fontsize=12, color="0.5",
    #     ha="left", va="top",
    #     )
    
    # # (0) Class predictions frequency -----------------------------------------
    
    # ax = axes[0]

    # for cl, cla in enumerate(main.ncl_classes):
    #     _barplot(
    #         ax, ncl_avg["freq"], sclasses, err=ncl_sem["freq"], 
    #         color=main.ncl_colors, 
    #         )
        
    # # Legend
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("Frequency")
    # ax.set_xticks(np.arange(0, len(sclasses)))
    # ax.set_xticklabels(sclasses, rotation=45, ha="center")
    # ax.set_title("Ncl. class pred. freq.", pad=10)
    
    # # (1) Class predictions entropy -------------------------------------------
    
    # ax = axes[1]
    
    # for cl, cla in enumerate(main.ncl_classes):
    #     vals = np.array(ncl_all[ncl_all["class"] == cla]["entropy"])
    #     _boxplot(ax, vals, cl, color=main.ncl_colors[cl])
        
    # # Legend
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("Entropy")
    # ax.set_xticks(np.arange(0, len(sclasses)))
    # ax.set_xticklabels(sclasses, rotation=45, ha="center")
    # ax.set_title("Ncl. class pred. entropy", pad=10)
    
    # # (3) Class ASC intensities -----------------------------------------------
    
    # ax = axes[3]
        
    # for t, tag in enumerate(["nsp", "spk"]):
    #     shift = -0.2 if t == 0 else 0.2
    #     for cl, cla in enumerate(main.ncl_classes):
    #         _barplot(
    #             ax, ncl_avg[f"asc_{tag}_int"], sclasses, 
    #             err=ncl_sem[f"asc_{tag}_int"],
    #             width=0.3, shift=shift, color=main.ncl_colors, 
    #             )
            
    # ax.text(
    #     0.5, 0.95, "non-speckle vs. speckle",
    #     fontsize=10, color="black",
    #     transform=ax.transAxes,
    #     horizontalalignment="center",
    #     verticalalignment="top",
    #     )
            
    # # Legend
    # _ylim(ax)
    # ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
    # ax.set_xticks(np.arange(0, len(sclasses)))
    # ax.set_xticklabels(sclasses, rotation=45, ha="center")
    # if main.int_norm:
    #     norm_str = "\n norm. int."
    # ax.set_title(f"Ncl. ASC int.{norm_str}", pad=10)
    
    # # (4) ASC distance class enrichment ---------------------------------------
    
    # ax = axes[4]
    
    # for cl, cla in enumerate(main.ncl_classes):
    #     x = dst_avg["bins"]
    #     vals = dst_avg[cla + "_asc_nsp_dst"]
    #     _lineplot(ax, vals, x, main.ncl_colors[cl], cla)
        
    # # Legend
    # ax.set_ylabel("Fold change")
    # ax.set_xlabel("ASC cluster dist. (pixels)")
    # ax.set_xticks(x)
    # if main.int_norm:
    #     norm_str = "\n norm. dist."
    # ax.set_title(f"Non-speckle dist. to ncl.{norm_str}", pad=10)
    
    # # (5) ASC distance class enrichment ---------------------------------------
    
    # ax = axes[5]
    
    # for cl, cla in enumerate(main.ncl_classes):
    #     x = dst_avg["bins"]
    #     vals = dst_avg[cla + "_asc_spk_dst"]
    #     _lineplot(ax, vals, x, main.ncl_colors[cl], cla)
        
    # # Legend
    # ax.set_ylabel("Fold change")
    # ax.set_xlabel("ASC cluster dist. (pixels)")
    # ax.set_xticks(x)
    # if main.int_norm:
    #     norm_str = "\n norm. dist."
    # ax.set_title(f"Speckle dist. to ncl.{norm_str}", pad=10)
    
    # # (6 - 11) Channel intensities --------------------------------------------
    
    # for cl, cla in enumerate(main.ncl_classes):

    #     ax = axes[cl + 6]                   
        
    #     for ch, chn in enumerate(schannels):
    #         vals = np.array(ncl_all[ncl_all["class"] == cla][f"{chn}_int"])
    #         _boxplot(ax, vals, ch, color=main.ncl_colors[cl])
                    
    #     # Legend
    #     _ylim(ax)
    #     ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
    #     ax.set_xticklabels(schannels, rotation=45, ha="center")
    #     if main.int_norm:
    #         norm_str = "\nnorm. int."
    #     ax.set_title(cla + norm_str, pad=10)
    
    # # -------------------------------------------------------------------------
    
    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    # # plt.close(fig) 
    # # plt.ion()