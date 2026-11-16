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
    get_ncl_intensities, get_ncl_distances, get_asc_stats, 
    format_ncl_data, format_asc_data, average_data, average_asc_stats,
    bin_ncl_distances, average_ncl_distances_bin,
    )

# plot
from plot import Plot

# bdtools
from bdtools.model.model import Model

# scipy
from scipy.stats import entropy

#%% Comments ------------------------------------------------------------------

#%% Inputs --------------------------------------------------------------------

main_parameters = {

    # Procedure
    "force_extract"         : 1,
    "force_predict_nuclei"  : 1,
    "force_predict_asc"     : 1,
    "force_classify_nuclei" : 1,
    "force_classify_asc"    : 1,
    "force_measure"         : 1,
    "force_analyse"         : 1,
    
    # Paths
    "data_path"             : Path("D:\local_mbl\data"),
    "model_pvw_name"        : "model-sm_preview_256_binary_227-1024",
    "model_ncl_name"        : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"        : "model-sm_asc_256_binary_377-1024",
    "model_ncl_class_name"  : "model-cls_nuclei_12_2_6_5045-8190b",
    "model_asc_class_name"  : "model-cls_asc_48_4_3_4467-8190b",
    
    # extract()
    "pvw_level"             : 4,
    "stk_level"             : 1,
        
    # predict_nuclei()
    "ncl_level"             : 0,
    "ncl_thresh_0"          : 0.20,
    "ncl_thresh_1"          : 0.40,
    "ncl_min_size"          : 8,    # in µm² at level = 0
    "ncl_max_size"          : 512,  # in µm² at level = 0
    
    # predict_asc()
    "asc_level"             : 0,
    "asc_thresh_0"          : 0.1,
    "asc_thresh_1"          : None,
    "asc_min_size"          : 8,    # in µm² at level = 0
    "asc_max_size"          : None, # in µm² at level = 0
    
    # Measure()
    "pixel_size"            : 0.5,  # in µm at level = 0
    "int_crop_size"         : 16,   # in µm at level = 0
    "int_norm"              : True, # normalize fluo. int. (per channel)
    "dst_bin_size"          : 4,    # in µm at level = 0
    "dst_bin_max"           : 32,   # in µm at level = 0
    "dst_norm"              : True, # normalize distances (per class)
    
    # Analyse()
    "skip_views" : [
        "251127_D2_S2_Scan1_unmixed_01",
        ], 
    
    # Maps  -------------------------------------------------------------------
    
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF",
        ],
    "ncl_classes" : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell",  
        ],
    "asc_classes" : [
        "Speckle", "Non-speckle", "Non-valid",
        ],
    
    "ncl_colors" : [
        [0.0, 0.0, 1.0], # Other
        [0.0, 0.8, 0.2], # Epithelium
        [1.0, 0.0, 0.0], # Neutrophil
        [1.0, 0.0, 1.0], # Macrophage
        [0.0, 1.0, 1.0], # Bcell
        [0.8, 0.8, 0.0], # Tcell
        ],

    # Plot  -------------------------------------------------------------------
    
    # Channels & classes
    "plot_channels" : ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"],
    "plot_classes"  : ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"],
    
    # Error bars
    "err_type" : "sem",
    
    # Line & grids
    "hline_params" : {
        "y" : 1, "color" : "0.75", "zorder" : 0,
        "linewidth" : 0.5, "linestyle" : ":",
        },
    
    "grid_params" : {
        "axis" : "x", "color" : "0.75", "zorder" : 0,
        "linewidth" : 0.5, "linestyle" : ":", 
        },

    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.initialize()
        if __name__ == "__main__":
            self.extract()
            self.predict_nuclei()
            self.predict_asc()
            self.classify_nuclei()
            self.classify_asc()
            self.measure()
            self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Log
        t0i = time.time()
        log_str = "Main() : initialize()"
        print("-" * len(log_str))
        print(log_str)
        print("-" * len(log_str))
        
        qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.qpt_paths, self.mtd_paths, self.vws_paths  = [], [], []
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
                
        t1f = time.time()
        print(f"> {t1f - t0i:.3f}s")
        
#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        for p, qpt_path in enumerate(self.qpt_paths): 

            # Metadata
            mtd_path = self.mtd_paths[p]
            with open(mtd_path, "rb") as file:
                mtd = pickle.load(file)
                
            # Data ------------------------------------------------------------
                    
            for roi_idx in range(len(mtd["roi"])):
                
                # Paths
                vws_name = str(qpt_path.name).replace(".qptiff", f"_{roi_idx:02d}")
                vws_path = self.data_path / vws_name                
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
            pvw_msk_path = vws_path / "preview_mask.tif"
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
                pvw_msk = io.imread(pvw_msk_path)
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
                    ncl_prd, pvw_msk,
                    thresh_0=self.ncl_thresh_0, 
                    thresh_1=self.ncl_thresh_1,
                    min_size=self.ncl_min_size,
                    max_size=self.ncl_max_size,
                    prd_level=self.ncl_level,
                    msk_level=self.stk_level,
                    pvw_level=self.pvw_level,
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
            pvw_msk_path = vws_path / "preview_mask.tif"
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
                pvw_msk = io.imread(pvw_msk_path)
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
                    asc_prd, pvw_msk,
                    thresh_0=self.asc_thresh_0, 
                    thresh_1=self.asc_thresh_1,
                    min_size=self.asc_min_size,
                    max_size=self.asc_max_size,
                    prd_level=self.asc_level,
                    msk_level=self.stk_level,
                    pvw_level=self.pvw_level,
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
            
            ncl_dst_bin_path = vws_path / "nuclei_distances_bin.csv"
            asc_sts_path = vws_path / "asc_stats.csv"
            ncl_data_path = vws_path / "nuclei_data.csv"
            ncl_data_avg_path = vws_path / "nuclei_data_avg.csv"
            ncl_data_std_path = vws_path / "nuclei_data_std.csv"
            ncl_data_sem_path = vws_path / "nuclei_data_sem.csv"
            asc_data_path = vws_path / "asc_data.csv"
            asc_data_avg_path = vws_path / "asc_data_avg.csv"
            asc_data_std_path = vws_path / "asc_data_std.csv"
            asc_data_sem_path = vws_path / "asc_data_sem.csv"
            
            plot_path = vws_path / "plot.png"
            
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
                    stk_nrm, ncl_prp, asc_msk, self.channels, 
                    crop_size=self.int_crop_size, normalize=self.int_norm,
                    )

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei intensities")
                                                        
                # Get nuclei distances (to asc) -------------------------------
                    
                t0 = time.time()
                                   
                ncl_dst = get_ncl_distances(
                    asc_msk, ncl_prp, stk_level=self.stk_level)
                
                ncl_dst_bin = bin_ncl_distances(
                    ncl_dst, ncl_cls, self.ncl_classes,
                    bin_size=self.dst_bin_size, 
                    bin_max=self.dst_bin_max, 
                    normalize=self.dst_norm, 
                    ) 
                
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
                
                # Plot --------------------------------------------------------
                
                plot = Plot(
                    self, 
                    ncl_data, ncl_data_avg, ncl_data_std, ncl_data_sem,
                    asc_data, asc_data_avg, asc_data_std, asc_data_sem,
                    ncl_dst_bin, asc_sts,
                    asc_sts_avg=None,
                    asc_sts_std=None,
                    asc_sts_sem=None,
                    merged=False,
                    )
                
                # Save --------------------------------------------------------
                                
                t0 = time.time()
                
                with open(ncl_int_path, "wb") as file:
                    pickle.dump(ncl_int, file)
                with open(ncl_dst_path, "wb") as file:
                    pickle.dump(ncl_dst, file)
                
                ncl_dst_bin.to_csv(ncl_dst_bin_path, index=False)
                asc_sts.to_csv(asc_sts_path, index=False)
                ncl_data.to_csv(ncl_data_path, index=False)
                ncl_data_avg.to_csv(ncl_data_avg_path, index=False)
                ncl_data_std.to_csv(ncl_data_std_path, index=False)
                ncl_data_sem.to_csv(ncl_data_sem_path, index=False)
                asc_data.to_csv(asc_data_path, index=False)
                asc_data_avg.to_csv(asc_data_avg_path, index=False)
                asc_data_std.to_csv(asc_data_std_path, index=False)
                asc_data_sem.to_csv(asc_data_sem_path, index=False)
                
                plot.fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                
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
            asc_sts_avg_path = self.data_path / "asc_stats_avg.csv"
            asc_sts_std_path = self.data_path / "asc_stats_std.csv"
            asc_sts_sem_path = self.data_path / "asc_stats_sem.csv"
            ncl_dst_bin_avg_path = self.data_path / "nuclei_distances_bin_avg.csv"
            ncl_dst_bin_std_path = self.data_path / "nuclei_distances_bin_std.csv"
            ncl_dst_bin_sem_path = self.data_path / "nuclei_distances_bin_sem.csv"
            
            plot_path = self.data_path / "plot_merged.png"
            
            # Load & format/avg. data -----------------------------------------
            
            t0 = time.time()
            
            ncl_data_avg_mrg, asc_data_avg_mrg, asc_sts_mrg = [], [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                if not vws_path.name in self.skip_views:
                
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
                        
            # Class avg. data
            ncl_data_avg_avg, ncl_data_avg_std, ncl_data_avg_sem = average_data(
                ncl_data_avg_mrg, self.ncl_classes, vws_path)
            asc_data_avg_avg, asc_data_avg_std, asc_data_avg_sem = average_data(
                asc_data_avg_mrg, self.asc_classes, vws_path)
            asc_sts_avg, asc_sts_std, asc_sts_sem = average_asc_stats(asc_sts_mrg)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - load & format/avg. data")

            # Bin ncl distances -----------------------------------------------
            
            t0 = time.time()
            
            ncl_dst_bin_mrg = []
            for p, vws_path in enumerate(self.vws_paths):
                
                if not vws_path.name in self.skip_views:
                
                    # Paths
                    ncl_dst_bin_path = vws_path / "nuclei_distances_bin.csv"
                    
                    # Load
                    ncl_dst_bin_mrg.append(pd.read_csv(ncl_dst_bin_path))
                
            ncl_dst_bin_avg, ncl_dst_bin_std, ncl_dst_bin_sem = (
                average_ncl_distances_bin(ncl_dst_bin_mrg))   
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - bin ncl distances")
            
            # Plot ------------------------------------------------------------
            
            plot = Plot(
                self, 
                ncl_data_avg_mrg, ncl_data_avg_avg, 
                ncl_data_avg_std, ncl_data_avg_sem,
                asc_data_avg_mrg, asc_data_avg_avg, 
                asc_data_avg_std, asc_data_avg_sem,
                ncl_dst_bin_avg, asc_sts_mrg,
                asc_sts_avg=asc_sts_avg,
                asc_sts_std=asc_sts_std,
                asc_sts_sem=asc_sts_sem,
                merged=True,
                )
            
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
            asc_sts_avg.to_csv(asc_sts_avg_path, index=False)
            asc_sts_std.to_csv(asc_sts_std_path, index=False)
            asc_sts_sem.to_csv(asc_sts_sem_path, index=False)
            ncl_dst_bin_avg.to_csv(ncl_dst_bin_avg_path)
            ncl_dst_bin_std.to_csv(ncl_dst_bin_std_path)
            ncl_dst_bin_sem.to_csv(ncl_dst_bin_sem_path)
            
            plot.fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
   
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)