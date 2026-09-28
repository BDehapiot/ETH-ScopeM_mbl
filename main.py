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
    get_nuclei_stats, format_nuclei_stats,
    get_asc_stats, format_asc_stats,
    get_asc_distances, format_asc_distances,
    get_obj_crops, get_crops_processed,
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
    "force_predict_nuclei"  : 1,
    "force_predict_asc"     : 1,
    "force_predict_classes" : 0,
    "force_measure"         : 0,
    "force_analyse"         : 0,
    
    # Paths
    "data_path"             : Path("D:\local_mbl\data"),
    "model_preview_name"    : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"     : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"        : "model-sm_asc_256_binary_307-1024",
    "model_classes_name"    : "model-cls_classes_12_2_6_5045-8190b",
    
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
            asc_msk_path = vws_path / "asc_mask.tif"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
            asc_sts_path = vws_path / "asc_stats.pkl"
            asc_dst_path = vws_path / "asc_distances.pkl"
            asc_spk_dst_path = vws_path / "asc_speckles_distances.pkl"

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
                asc_msk = io.imread(asc_msk_path)
                
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
                                
                asc_sts = get_asc_stats(
                    pvw_msk, ncl_ctd, asc_msk, 
                    pvw_level=self.pvw_level, 
                    stk_level=self.stk_level, 
                    )
                
                asc_dst = get_asc_distances(
                    asc_msk > 0, ncl_ctd, self.classes,
                    stk_level=self.stk_level
                    )
                
                asc_spk_dst = get_asc_distances(
                    asc_msk > 200, ncl_ctd, self.classes,
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
                with open(asc_spk_dst_path, "wb") as file:
                    pickle.dump(asc_spk_dst, file)
                    
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
        aasc_spk_dst_df_path = self.data_path / "averaged_asc_speckles_dst_df.csv"
        aasc_spk_dst_dfn_path = self.data_path / "averaged_asc_speckles_dst_dfn.csv"
        mncl_plot_path = self.data_path / "merged_nuclei_plot.png"
        
        if ((self.force_analyse > 0 and not mncl_dfan_path.exists()) 
            or self.force_analyse == 2):     
        
            mncl_df, mncl_dfa, mncl_dfn, mncl_dfan = [], [], [], []
            masc_sts_df, aasc_dst_df, aasc_dst_dfn, aasc_spk_dst_df, aasc_spk_dst_dfn = [], [], [], [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
                ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
                ncl_sts_path = vws_path / "nuclei_stats.pkl"
                asc_sts_path = vws_path / "asc_stats.pkl"
                asc_dst_path = vws_path / "asc_distances.pkl"
                asc_spk_dst_path = vws_path / "asc_speckles_distances.pkl"
                ncl_df_path = vws_path / "nuclei_df.csv"
                ncl_dfa_path = vws_path / "nuclei_dfa.csv"
                ncl_dfn_path = vws_path / "nuclei_dfn.csv"
                ncl_dfan_path = vws_path / "nuclei_dfan.csv"
                asc_sts_df_path = vws_path / "asc_sts_df.csv"
                asc_dst_df_path = vws_path / "asc_dst_df.csv"
                asc_dst_dfn_path = vws_path / "asc_dst_dfn.csv"
                asc_spk_dst_df_path = vws_path / "asc_spk_dst_df.csv"
                asc_spk_dst_dfn_path = vws_path / "asc_spk_dst_dfn.csv"
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
                with open(asc_spk_dst_path, "rb") as file:
                    asc_spk_dst = pickle.load(file)
                
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
                
                asc_spk_dst_df, asc_spk_dst_dfn = format_asc_distances(
                    asc_spk_dst, ncl_prd, self.classes, stk_level=self.stk_level,
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
                asc_spk_dst_df.to_csv(asc_spk_dst_df_path, index=False)
                asc_spk_dst_dfn.to_csv(asc_spk_dst_dfn_path, index=False)
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
                aasc_spk_dst_df.append(asc_spk_dst_df)
                aasc_spk_dst_dfn.append(asc_spk_dst_dfn)
                
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
            aasc_spk_dst_df = pd.concat(aasc_spk_dst_df).groupby(level=0).mean()
            aasc_spk_dst_dfn = pd.concat(aasc_spk_dst_dfn).groupby(level=0).mean()
            
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
            aasc_spk_dst_df.to_csv(aasc_spk_dst_df_path, index=False)
            aasc_spk_dst_dfn.to_csv(aasc_spk_dst_dfn_path, index=False)
            fig.savefig(mncl_plot_path, dpi=150, bbox_inches="tight")
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)    
    
#%%

    def get_mask_from_props(prp, shape=(2048, 2048)):
        msk = np.zeros(shape, dtype=bool)
        for coords in prp["coords"]:
            msk[tuple(coords.T)] = True
        return msk

    # -------------------------------------------------------------------------    

    # Paths
    vws_path = main.vws_paths[0]
    stk_nrm_path = vws_path / "stack_norm.tif"
    ncl_prp_path = vws_path / "nuclei_properties.pkl"
    
    # Load
    stk_nrm = io.imread(stk_nrm_path)
    with open(ncl_prp_path, "rb") as file:
        ncl_prp = pickle.load(file)
        
    # Get mask
    ncl_msk = get_mask_from_props(ncl_prp, shape=stk_nrm[0, ...].shape)
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(ncl_msk, colormap="gray")
       
#%% 

    # stk_nrm_path = main.stk_nrm_path
    # ncl_msk = main.ncl_msk

    # import tifffile
    # from skimage.measure import label, regionprops_table

    # def imread_channel(path, channel_idx=0, normalize=True):
    #     img_memmap = tifffile.memmap(path)
    #     img = img_memmap[channel_idx, :, :]
    #     if normalize:
    #         return img.astype("float32") / 255
    #     else:
    #         return img
    
    # def get_obj_properties(nrm, msk):

    #     props = regionprops_table(
    #         label(msk), intensity_image=nrm, 
    #         properties=(
    #             "area", "coords", "centroid", "eccentricity",
    #             "intensity_max", "intensity_min",
    #             "intensity_mean", "intensity_std",
    #             ),
    #         )

    #     return {
    #         "y"      : list(props["centroid-0"].astype("uint16")),  
    #         "x"      : list(props["centroid-1"].astype("uint16")),  
    #         "area"   : list(props["area"].astype("uint16")), 
    #         "coords" : [coord.astype("uint16") for coord in props["coords"]], 
    #         "eccentricity"   : list(props["eccentricity"].astype("float32")), 
    #         "intensity_max"  : list(props["intensity_max"].astype("float32")),  
    #         "intensity_min"  : list(props["intensity_min"].astype("float32")), 
    #         "intensity_mean" : list(props["intensity_mean"].astype("float32")), 
    #         "intensity_std"  : list(props["intensity_std"].astype("float32")), 
    #         }       
            
    # ncl_nrm = imread_channel(stk_nrm_path, channel_idx=0)
    # ncl_prp = get_obj_properties(ncl_nrm, ncl_msk)
    
    # ncl_prp_path = Path("ncl_prp.pkl")
    # with open(ncl_prp_path, "wb") as file:
    #     pickle.dump(ncl_prp, file)
    
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     rescale(ncl_nrm, rf), colormap="gray",
    #     gamma=0.5, opacity=0.75,
    #     )
    # vwr.add_image(
    #     ncl_msk, colormap="bop orange",
    #     blending="additive", opacity=0.5,
    #     )
    
#%%
    
    # # bdtools
    # from bdtools.conn import lbl_conn

    # # skimage
    # from skimage.transform import rescale
    # from skimage.segmentation import watershed
    # from skimage.morphology import remove_small_objects
    # from skimage.measure import label, regionprops_table

    # def get_obj_mask(
    #         prd, thresh_0=0.25, thresh_1=0.50, min_size=8, 
    #         prd_level=0, msk_level=1,
    #         ):
        
    #     # Initialize
    #     rf = 1 / (2 ** (msk_level - prd_level))
    #     min_size = min_size // ((2 ** msk_level) ** 2) 
        
    #     # Rescale prediction (if required)
    #     if rf != 1: 
    #         prd = rescale(prd, rf)
            
    #     # Get mask
    #     if thresh_1 is not None:
    #         msk_0 = prd > thresh_0
    #         msk_1 = prd > thresh_1
    #         msk = watershed(-prd, label(msk_1), mask=msk_0)
    #         msk[lbl_conn(msk, conn=2) == 2] = 0
    #         msk = msk > 0
    #     else:
    #         msk = prd > thresh_0
        
    #     # Remove small objects
    #     if min_size > 0:
    #         msk = remove_small_objects(msk, min_size=min_size)
        
    #     return msk
    
    # def get_obj_centroids(msk, n_levels=5, msk_level=0):
    #     props = regionprops_table(label(msk), properties=("label", "centroid"))
    #     y, x = props["centroid-0"], props["centroid-1"]
    #     ctd = {"y" : [], "x" : []}
    #     for level in range(n_levels):
    #         factor = 2 ** (msk_level - level)
    #         ctd["y"].append(y * factor)
    #         ctd["x"].append(x * factor)
    #     return ctd

    # ncl_msk = get_obj_mask(
    #     ncl_prd, 
    #     thresh_0=main_parameters["ncl_thresh_0"], 
    #     thresh_1=main_parameters["ncl_thresh_1"], 
    #     min_size=main_parameters["ncl_min_size"],
    #     prd_level=main_parameters["ncl_level"],
    #     )
    
    # ncl_ctd = get_obj_centroids(
    #     ncl_msk, 
    #     n_levels=len(mtd["level"]), 
    #     msk_level=main_parameters["ncl_level"],
    #     )
        
    # # Display
    # import napari
    # rf = 1 / (2 ** (main_parameters["stk_level"] - main_parameters["ncl_level"]))
    
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     rescale(ncl_nrm, rf), colormap="gray",
    #     gamma=0.5, opacity=0.75,
    #     )
    # vwr.add_image(
    #     ncl_msk, colormap="bop orange",
    #     blending="additive", opacity=0.5,
    #     )
   
    
#%%
    
    # data_path = main_parameters["data_path"]
    # vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    
    # # -------------------------------------------------------------------------
    
    # # Paths
    # ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    # pvw_msk_path = vws_path / "preview_mask.tif"  
    # stk_nrm_path = vws_path / "stack_norm.tif"  
    # asc_msk_path = vws_path / "asc_mask.tif"
    # ncl_sts_path = vws_path / "nuclei_stats.pkl"
    # asc_sts_path = vws_path / "asc_stats.pkl"
    # asc_dst_path = vws_path / "asc_distances.pkl"
    
    # ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
    
    # # -------------------------------------------------------------------------
    
    # with open(ncl_ctd_path, "rb") as file:
    #     ncl_ctd = pickle.load(file)
    # with open(ncl_prd_path, "rb") as file:
    #     ncl_prd = pickle.load(file)
    # pvw_msk = io.imread(pvw_msk_path)
    # stk_nrm = io.imread(stk_nrm_path)
    # asc_msk = io.imread(asc_msk_path)
        
    # # -------------------------------------------------------------------------
    
    # from skimage.measure import label
    # from scipy.ndimage import distance_transform_edt
    
    # def get_asc_stats(
    #         pvw_msk, ncl_ctd, asc_msk,
    #         pvw_level=4, stk_level=1
    #         ):
        
    #     # Init.
    #     asc_sts = {}
    #     pvw_factor = (2 ** pvw_level) ** 2
    #     asc_factor = (2 ** stk_level) ** 2
        
    #     # Areas
    #     all_area = np.sum(pvw_msk > 0) * pvw_factor
    #     asc_area = np.sum(asc_msk > 0) * asc_factor
    #     asc_area_nrm = asc_area / all_area
        
    #     # Counts
    #     ncl_num = len(ncl_ctd["y"][0])
    #     asc_num = np.max(label(asc_msk > 0))
    #     asc_spk_num = np.max(label(asc_msk > 200))
    #     asc_num_ratio = asc_num / ncl_num
    #     asc_spk_num_ratio = asc_spk_num / ncl_num
                
    #     # Append
    #     asc_sts["all_area"         ] = all_area
    #     asc_sts["asc_area"         ] = asc_area
    #     asc_sts["asc_area_ratio"   ] = asc_area_nrm
    #     asc_sts["ncl_num"          ] = ncl_num
    #     asc_sts["asc_num"          ] = asc_num
    #     asc_sts["asc_spk_num"      ] = asc_spk_num
    #     asc_sts["asc_num_ratio"    ] = asc_num_ratio
    #     asc_sts["asc_spk_num_ratio"] = asc_spk_num_ratio
        
    #     return asc_sts
    
    # def format_asc_stats(asc_sts, vws_path):
    #     asc_sts_df = pd.DataFrame([asc_sts])
    #     asc_sts_df.insert(0, "view", vws_path.name)
    #     return asc_sts_df
    
    # def get_asc_distances(asc_msk, ncl_ctd, classes, stk_level=1):
                
    #     # Get edt
    #     asc_edt = distance_transform_edt(~asc_msk)
        
    #     # Get asc distances
    #     coords = (
    #         ncl_ctd["y"][stk_level].astype(int), 
    #         ncl_ctd["x"][stk_level].astype(int),
    #         )
    #     asc_dst = asc_edt[coords]
        
    #     # Scale acc. to stk_level
    #     asc_dst *= 2 ** stk_level
        
    #     return asc_dst
    
    # def format_asc_distances(
    #     asc_dst, ncl_prd, classes, bin_size=10, bin_max=200, stk_level=1):
        
    #     def get_counts(ncl_prd, normalized=False):
    #         _, counts = np.unique(ncl_prd, return_counts=True)
    #         if normalized:
    #             counts = counts.astype("float32") / np.sum(counts)
    #         return counts
        
    #     # Get bins
    #     bins = np.arange(bin_size, bin_max + bin_size, bin_size)
        
    #     # Count nuclei class
    #     counts_raw = []
    #     counts_all_raw = np.stack(get_counts(ncl_prd, normalized=True))
    #     for d, dst in enumerate(bins):
    #         dst0 = 0 if d == 0 else bins[d - 1]
    #         ncl_prd_dst = ncl_prd[(asc_dst >= dst0) & (asc_dst < dst)]
    #         counts_raw.append(get_counts(ncl_prd_dst, normalized=True))
            
    #     # Normalizations (enrichment)
    #     counts_nrm = []
    #     for d, dst in enumerate(bins):
    #         counts_nrm.append(counts_raw[d] / counts_all_raw)
        
    #     # # Get dataframe
    #     asc_dst_df = pd.DataFrame(counts_raw, columns=classes)
    #     asc_dst_df.insert(0, "distances", bins)
    #     asc_dst_dfn = pd.DataFrame(counts_nrm, columns=classes)
    #     asc_dst_dfn.insert(0, "distances", bins)
        
    #     return asc_dst_df, asc_dst_dfn
    
    # # -------------------------------------------------------------------------
    
    # asc_sts = get_asc_stats(
    #     pvw_msk, ncl_ctd, asc_msk, 
    #     pvw_level=4, stk_level=1
    #     )
    
    # asc_sts_df = format_asc_stats(asc_sts, vws_path)
    
    # classes = main_parameters["classes"]
    
    # asc_dst = get_asc_distances(asc_msk > 200, ncl_ctd, classes, stk_level=1)
    
    # asc_dst_df, asc_dst_dfn = format_asc_distances(
    #     asc_dst, ncl_prd, classes, bin_size=10, bin_max=200, stk_level=1)