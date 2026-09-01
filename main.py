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
    norm_data, 
    get_slide_metadata, load_view, 
    get_preview_processed, get_preview_stats,
    get_nuclei_mask, get_nuclei_centroids, 
    get_nuclei_stats, format_nuclei_stats,
    get_asc_processed, get_asc_stats, format_asc_stats,
    get_asc_distances, format_asc_distances,
    get_crops, get_crops_processed,
    plot,
    )

# bdtools
from bdtools.model.model import Model

# scipy
from scipy.stats import entropy

#%% Inputs --------------------------------------------------------------------

main_parameters = {

    # Procedure
    "force_extract"         : 0,
    "force_predict_nuclei"  : 0,
    "force_predict_asc"     : 0,
    "force_predict_classes" : 0,
    "force_measure"         : 0,
    "force_analyse"         : 0,
    
    # Paths
    "data_path"             : Path("D:\local_mbl\data"),
    "model_preview_name"    : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"     : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"        : "model-sm_asc_256_edt_307-1024",
    "model_classes_name"    : "model-cls_classes_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level"             : 4,
    "stk_level"             : 1,
        
    # predict()
    "ncl_level"             : 0,
    "ncl_thresh_0"          : 0.25,
    "ncl_thresh_1"          : 0.50,
    "asc_level"             : 0,
    "asc_thresh"            : 0.05,
    
    # Measure()
    "crop_size"             : 64, # in pixels at stk_level = 0
    
    # Analyse()
    "asc_bin_size"          : 8,  # in pixels at stk_level = 0
    "asc_bin_max"           : 64, # in pixels at stk_level = 0
    
    # Maps  -------------------------------------------------------------------
    
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF",
        ],
    "classes"  : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell",  
        ],
    
    "colors" : [
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
        self.predict_classes()
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
                            model_path=Path.cwd() / self.model_preview_name
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
            ncl_msk_path = vws_path / "nuclei_mask.tif"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            roi_idx = int(vws_path.stem[-2:])

            if ((self.force_predict_nuclei > 0 and not ncl_ctd_path.exists()) 
                or self.force_predict_nuclei == 2):
                       
                # Log
                t0i = time.time()
                log_str = f"Main() : predict_nuclei() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_nuclei_path = Path.cwd() / self.model_nuclei_name
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, model_path=model_nuclei_path)
                                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                ncl_raw = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=self.ncl_level, roi_idx=roi_idx,
                    )
                ncl_nrm = norm_data(ncl_raw, pvw_sts, channel="DAPI")
                
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
    
                ncl_msk = get_nuclei_mask(
                    ncl_prd, 
                    thresh_0=self.ncl_thresh_0, 
                    thresh_1=self.ncl_thresh_1, 
                    ncl_level=self.ncl_level,
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get mask")
    
                # Get centroids -----------------------------------------------
    
                t0 = time.time()
                               
                ncl_ctd = get_nuclei_centroids(
                    ncl_msk, 
                    n_levels=len(mtd["level"]), 
                    ncl_level=self.ncl_level,
                    )

                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get centroids")
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_ctd_path, "wb") as file:
                    pickle.dump(ncl_ctd, file)
    
                # Clear
                del ncl_raw, ncl_nrm, ncl_prd, ncl_msk, ncl_ctd
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
            asc_prc_path = vws_path / "asc_processed.tif"
            roi_idx = int(vws_path.stem[-2:])
            
            if ((self.force_predict_asc > 0 and not asc_prc_path.exists()) 
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
                asc_raw = load_view(
                    qpt_path, metadata=mtd, channels="ASC", 
                    level=self.asc_level, roi_idx=roi_idx,
                    )
                asc_nrm = norm_data(asc_raw, pvw_sts, channel="ASC")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Predict -----------------------------------------------------
                
                t0 = time.time()

                asc_prd = model_asc.predict(
                    asc_nrm, patch_overlap=32, batch_size=32, chunk_size=5000)
                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - predict")
    
                # Process -----------------------------------------------------
    
                t0 = time.time()
    
                asc_prc = get_asc_processed(
                    asc_nrm, asc_prd,
                    thresh=self.asc_thresh,
                    asc_level=self.asc_level, 
                    stk_level=self.stk_level,
                    )
                    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - process")   
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                io.imsave(
                    asc_prc_path, 
                    (asc_prc * 255).astype("uint8"),
                    check_contrast=False,
                    )
                
                # Clear
                del asc_raw, asc_nrm, asc_prd, asc_prc
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : predict_classes() -------------------------------------------
            
    def predict_classes(self):
        
        for p, vws_path in enumerate(self.vws_paths):
            
            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            stk_nrm_path = vws_path / "stack_norm.tif"
            ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
            ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
            
            if ((self.force_predict_classes > 0 and not ncl_ent_path.exists()) 
                or self.force_predict_classes == 2):    
            
                # Log
                t0i = time.time()
                log_str = f"Main() : predict_classes() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))    
                
                # Model
                if "model_classes" not in locals():
                    model_classes = Model(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_classes_name
                        )
                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                stk_nrm = io.imread(stk_nrm_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Get crops ---------------------------------------------------
                    
                t0 = time.time()

                crop_size = int(self.model_classes_name.split("_")[2])
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=crop_size, 
                    stk_level=self.stk_level
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Predict -----------------------------------------------------

                t0 = time.time()

                crp_prc = get_crops_processed(crp_nrm)
                ncl_prd = model_classes.predict(crp_prc)
                ncl_ent = entropy(ncl_prd, axis=1, base=len(self.classes))
                ncl_prd = np.argmax(ncl_prd, axis=1)
    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - predict")    
    
                # Save --------------------------------------------------------
    
                t0 = time.time()
                
                with open(ncl_prd_path, "wb") as file:
                    pickle.dump(ncl_prd, file)
                with open(ncl_ent_path, "wb") as file:
                    pickle.dump(ncl_ent, file)
                    
                # Clear
                del stk_nrm, ncl_ctd, crp_nrm, crp_prc, ncl_prd, ncl_ent
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
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            pvw_msk_path = vws_path / "preview_mask.tif"  
            stk_nrm_path = vws_path / "stack_norm.tif"  
            asc_prc_path = vws_path / "asc_processed.tif"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
            asc_sts_path = vws_path / "asc_stats.pkl"
            asc_dst_path = vws_path / "asc_distances.pkl"

            if ((self.force_measure > 0 and not ncl_sts_path.exists()) 
                or self.force_measure == 2):                    

                # Log
                t0i = time.time()
                log_str = f"Main() : measure() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))                           
        
                # Load --------------------------------------------------------

                t0 = time.time()

                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                pvw_msk = io.imread(pvw_msk_path)
                stk_nrm = io.imread(stk_nrm_path)
                asc_prc = io.imread(asc_prc_path)
                stk_nrm[4, ...] = asc_prc # replace asc channel
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=self.crop_size // 2 ** self.stk_level, 
                    crop_fraction=None, 
                    stk_level=self.stk_level,
                    )
                
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Get nuclei stats --------------------------------------------
                    
                t0 = time.time()

                crp_nrm = crp_nrm.astype("float32") / 255 # normalize
                ncl_sts = get_nuclei_stats(crp_nrm, self.channels)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei stats")
                
                # Get asc data ------------------------------------------------
                    
                t0 = time.time()
                
                asc_prc = asc_prc.astype("float32") / 255 # normalize
                asc_sts = get_asc_stats(
                    pvw_msk, asc_prc, 
                    pvw_level=self.pvw_level, 
                    stk_level=self.stk_level, 
                    )
                
                asc_dst = get_asc_distances(
                    asc_prc, ncl_ctd, self.classes,
                    stk_level=self.stk_level
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get asc data")
                                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_sts_path, "wb") as file:
                    pickle.dump(ncl_sts, file)
                with open(asc_sts_path, "wb") as file:
                    pickle.dump(asc_sts, file)
                with open(asc_dst_path, "wb") as file:
                    pickle.dump(asc_dst, file)
                    
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                    
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
#%% Class(Main) : analyse() ---------------------------------------------------  

    def analyse(self):
        
        # Paths
        mncl_df_path = self.data_path / "merged_nuclei_df.csv"
        mncl_dfa_path = self.data_path / "merged_nuclei_dfa.csv"
        mncl_dfn_path = self.data_path / "merged_nuclei_dfn.csv"
        mncl_dfan_path = self.data_path / "merged_nuclei_dfan.csv"
        masc_sts_df_path = self.data_path / "merged_asc_sts_df.csv"
        aasc_dst_df_path = self.data_path / "averaged_asc_dst_df.csv"
        aasc_dst_dfn_path = self.data_path / "averaged_asc_dst_dfn.csv"
        mncl_plot_path = self.data_path / "merged_nuclei_plot.png"
        
        if ((self.force_analyse > 0 and not mncl_dfan_path.exists()) 
            or self.force_analyse == 2):     
        
            mncl_df, mncl_dfa, mncl_dfn, mncl_dfan = [], [], [], []
            masc_sts_df, aasc_dst_df, aasc_dst_dfn = [], [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
                ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
                ncl_sts_path = vws_path / "nuclei_stats.pkl"
                asc_sts_path = vws_path / "asc_stats.pkl"
                asc_dst_path = vws_path / "asc_distances.pkl"
                ncl_df_path = vws_path / "nuclei_df.csv"
                ncl_dfa_path = vws_path / "nuclei_dfa.csv"
                ncl_dfn_path = vws_path / "nuclei_dfn.csv"
                ncl_dfan_path = vws_path / "nuclei_dfan.csv"
                asc_sts_df_path = vws_path / "asc_sts_df.csv"
                asc_dst_df_path = vws_path / "asc_dst_df.csv"
                asc_dst_dfn_path = vws_path / "asc_dst_dfn.csv"
                ncl_plot_path = vws_path / "nuclei_plot.png"
    
                # Log
                t0i = time.time()
                log_str = f"Main() : analyse : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))   
                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_prd_path, "rb") as file:
                    ncl_prd = pickle.load(file)
                with open(ncl_ent_path, "rb") as file:
                    ncl_ent = pickle.load(file)
                with open(ncl_sts_path, "rb") as file:
                    ncl_sts = pickle.load(file)
                with open(asc_sts_path, "rb") as file:
                    asc_sts = pickle.load(file)
                with open(asc_dst_path, "rb") as file:
                    asc_dst = pickle.load(file)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                                    
                # Format data -------------------------------------------------
                
                t0 = time.time()
                
                ncl_df, ncl_dfa, ncl_dfn, ncl_dfan = format_nuclei_stats(
                    ncl_prd, ncl_ent, ncl_sts, vws_path, 
                    self.channels, self.classes
                    )
                
                asc_sts_df = format_asc_stats(asc_sts, vws_path)
                asc_dst_df, asc_dst_dfn = format_asc_distances(
                    asc_dst, ncl_prd, self.classes, stk_level=self.stk_level,
                    bin_size=self.asc_bin_size, bin_max=self.asc_bin_max, 
                    )
                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - format data")
                                      
                # Plot data ---------------------------------------------------
                
                t0 = time.time()
                
                fig = plot(
                    ncl_df, ncl_dfa, ncl_dfn, ncl_dfan, asc_dst_df, asc_dst_dfn,
                    self.asc_bin_size, self.asc_bin_max, 
                    self.channels, self.classes, self.colors
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - Plot data")
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                ncl_df.to_csv(ncl_df_path, index=False)
                ncl_dfa.to_csv(ncl_dfa_path, index=False)
                ncl_dfn.to_csv(ncl_dfn_path, index=False)
                ncl_dfan.to_csv(ncl_dfan_path, index=False)
                asc_sts_df.to_csv(asc_sts_df_path, index=False)
                asc_dst_df.to_csv(asc_dst_df_path, index=False)
                asc_dst_dfn.to_csv(asc_dst_dfn_path, index=False)
                fig.savefig(ncl_plot_path, dpi=150, bbox_inches="tight")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                # Append
                mncl_df.append(ncl_df)
                mncl_dfa.append(ncl_dfa)
                mncl_dfn.append(ncl_dfn)
                mncl_dfan.append(ncl_dfan)
                masc_sts_df.append(asc_sts_df)
                aasc_dst_df.append(asc_dst_df)
                aasc_dst_dfn.append(asc_dst_dfn)
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                    
            # Merge -----------------------------------------------------------
            
            # Log
            t0i = time.time()
            log_str = "Main() : analyse : merge"
            print("-" * len(log_str))
            print(log_str)
            print("-" * len(log_str)) 
            
            # Merge or average        
            mncl_df = pd.concat(mncl_df, axis=0, ignore_index=True)
            mncl_dfa = pd.concat(mncl_dfa, axis=0, ignore_index=True)
            mncl_dfn = pd.concat(mncl_dfn, axis=0, ignore_index=True)
            mncl_dfan = pd.concat(mncl_dfan, axis=0, ignore_index=True)
            masc_sts_df = pd.concat(masc_sts_df, axis=0, ignore_index=True)
            aasc_dst_df = pd.concat(aasc_dst_df).groupby(level=0).mean()
            aasc_dst_dfn = pd.concat(aasc_dst_dfn).groupby(level=0).mean()
            
            # Plot ------------------------------------------------------------
            
            t0 = time.time()
            
            fig = plot(
                mncl_df, mncl_dfa, mncl_dfn, mncl_dfan, aasc_dst_df, aasc_dst_dfn,
                self.asc_bin_size, self.asc_bin_max, 
                self.channels, self.classes, self.colors
                )
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - plot")
            
            # Save ------------------------------------------------------------
            
            t0 = time.time()
            
            mncl_df.to_csv(mncl_df_path, index=False)
            mncl_dfa.to_csv(mncl_dfa_path, index=False)
            mncl_dfn.to_csv(mncl_dfn_path, index=False)
            mncl_dfan.to_csv(mncl_dfan_path, index=False)
            masc_sts_df.to_csv(masc_sts_df_path, index=False)
            aasc_dst_df.to_csv(aasc_dst_df_path, index=False)
            aasc_dst_dfn.to_csv(aasc_dst_dfn_path, index=False)
            fig.savefig(mncl_plot_path, dpi=150, bbox_inches="tight")
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)    
    
#%%

    # Imports
    from skimage.transform import resize
    
    # Parameters
    vws_idx = 0
    y0, yn = 4000, 2000
    x0, xn = 3000, 2000
    y1 = y0 + yn
    x1 = x0 + xn
    
    # Fetch
    vws_path = main.vws_paths[vws_idx]
    pvw_msk_path = vws_path / "preview_mask.tif"
    stk_nrm_path = vws_path / "stack_norm.tif"
    asc_prc_path = vws_path / "asc_processed.tif"
    pvw_msk = io.imread(pvw_msk_path)
    stk_nrm = io.imread(stk_nrm_path)
    asc_prc = io.imread(asc_prc_path)
    asc_nrm = stk_nrm[4, ...]    
    
    # Prepare masks
    pvw_msk = resize(pvw_msk, asc_nrm.shape) > 0.5
    
    # Crop
    # asc_nrm = asc_nrm[y0:y1, x0:x1] 
    # asc_prc = asc_prc[y0:y1, x0:x1] 
    # pvw_msk = pvw_msk[y0:y1, x0:x1] 
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     asc_nrm, colormap="gray",
    #     blending="additive", opacity=0.5, gamma=0.25, 
    #     )
    # vwr.add_image(
    #     asc_msk, colormap="bop orange",
    #     blending="additive", opacity=0.1, gamma=1.0, 
    #     )
    
#%%

    import napari
    from skimage import draw
    from scipy.stats import pearsonr
    from skimage.measure import label
    from skimage.transform import rescale
    from skimage.feature import peak_local_max

    def get_asc_processed(asc_nrm, asc_prd, thresh=0.2, asc_level=0, stk_level=1):
        asc_msk = asc_prd > thresh
        asc_prc = asc_nrm.copy()
        asc_prc[~asc_msk] = 0 
        factor = 2 ** (stk_level - asc_level)
        asc_prc = rescale(asc_prc, 1 / factor)
        return asc_prc

    def get_asc_mask(
        asc_nrm, asc_prc,
        min_dist=5, min_int=25,
        radius=7, sigma=1.0, min_corr=0.9, 
        display=False,
        ):
        
        # Nested function(s) --------------------------------------------------
        
        def get_gaussian_kernel(radius, sigma):
            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            return kernel / kernel.max()
        
        def kernel_validation(
            arr, coords, radius=5, sigma=2.0, min_correlation=0.85):
            
            # Get gaussian kernel
            kernel = get_gaussian_kernel(radius, sigma)
            kernel_flat = kernel.ravel()
        
            valid = []
            r = radius
            h, w = arr.shape
            for y, x in coords:
                
                # Check boundaries
                if y - r < 0 or y + r >= h or x - r < 0 or x + r >= w:
                    continue
        
                # Patch and correlate
                patch = arr[y - r : y + r + 1, x - r : x + r + 1]
                corr, _ = pearsonr(patch.ravel(), kernel_flat)
                if corr >= min_correlation:
                    valid.append([y, x])
        
            return np.array(valid)
        
        # Execute -------------------------------------------------------------
        
        # Find local maxima
        coords = peak_local_max(
            asc_nrm, min_distance=min_dist, threshold_abs=min_int
            )
        
        # Validate candidates
        valid_coords = kernel_validation(
            asc_nrm, coords, radius=radius, sigma=sigma, 
            min_correlation=min_corr,
            )
        
        # Filter asc_msk
        asc_msk = (asc_prc > 0).astype(np.uint8)
        asc_lbl = label(asc_msk)
        if len(valid_coords) > 0:
            valid_lbl = np.unique(asc_lbl[valid_coords[:, 0], valid_coords[:, 1]])
            valid_lbl = valid_lbl[valid_lbl != 0]
            asc_msk[np.isin(asc_lbl, valid_lbl)] = 2
        
        # Make display
        if display:
            
            import napari
            
            asc_display = np.zeros_like(asc_nrm, dtype=bool)
            for y, x in coords:
                rr, cc = draw.disk((y, x), 1, shape=asc_nrm.shape)
                asc_display[rr, cc] = True
            for y, x in valid_coords:
                rr, cc = draw.circle_perimeter(y, x, radius, shape=asc_nrm.shape)
                asc_display[rr, cc] = True
                
            vwr = napari.Viewer()
            vwr.add_image(
                asc_nrm, colormap="gray",
                blending="additive", opacity=1.0, gamma=0.5, 
                )
            vwr.add_image(
                asc_display, colormap="yellow",
                blending="additive", opacity=0.75, gamma=1.0,
                )
            vwr.add_labels(
                asc_msk, 
                blending="additive", opacity=0.25,
                )
            
        return asc_msk
    
    # -------------------------------------------------------------------------   
    
    # Parameters
    parameters = {
        "min_dist" : 5,
        "min_int"  : 20,
        "radius"   : 7,
        "sigma"    : 0.75,
        "min_corr" : 0.85,
        "display"  : True,
        }

    t0 = time.time()
    
    asc_msk = get_asc_mask(
        asc_nrm, asc_prc, **parameters
        )
    
    t1 = time.time()
    print(f"get_asc_mask() : {t1 - t0:.3f}s")