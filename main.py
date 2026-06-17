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
    get_nuclei_mask, get_nuclei_centroids, get_nuclei_stats, format_nuclei_stats,
    get_asc_processed, get_asc_distances, format_asc_distances,
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
    "force_extract"         : 1,
    "force_predict_nuclei"  : 1,
    "force_predict_asc"     : 1,
    "force_predict_classes" : 1,
    "force_measure"         : 1,
    "force_analyse"         : 2,
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"     : "model-sm_asc_256_edt_243-1024",
    "model_classes_name" : "model-cls_classes_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level"     : 4,
    "stk_level"     : 1,
        
    # predict()
    "ncl_level"     : 0,
    "ncl_thresh_0"  : 0.25,
    "ncl_thresh_1"  : 0.50,
    "asc_level"     : 0,
    "asc_thresh"    : 0.05,
    
    # Measure()
    "crop_size"     : 64,  # in pixels at stk_level = 0
    
    # Analyse()
    "asc_bin_size"  : 10,  # in pixels at stk_level = 0
    "asc_bin_max"   : 200, # in pixels at stk_level = 0
    
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

                crop_size = int(self.model_classes_name.split("_")[1])
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
            stk_nrm_path = vws_path / "stack_norm.tif"  
            asc_prc_path = vws_path / "asc_processed.tif"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
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
                crp_nrm = crp_nrm.astype("float32") / 255 # normalize
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Get nuclei stats --------------------------------------------
                    
                t0 = time.time()

                ncl_sts = get_nuclei_stats(crp_nrm, self.channels)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get nuclei stats")
                
                # Get asc distances -------------------------------------------
                    
                t0 = time.time()
                
                asc_dst = get_asc_distances(
                    asc_prc, ncl_ctd, self.classes,
                    stk_level=self.stk_level
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get asc distances")
                    
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_sts_path, "wb") as file:
                    pickle.dump(ncl_sts, file)
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
        aasc_df_path = self.data_path / "averaged_asc_df.csv"
        aasc_dfn_path = self.data_path / "averaged_asc_dfn.csv"
        mncl_plot_path = self.data_path / "merged_nuclei_plot.png"
        
        if ((self.force_analyse > 0 and not mncl_dfan_path.exists()) 
            or self.force_analyse == 2):     
        
            aasc_df, aasc_dfn = [], []
            mncl_df, mncl_dfa, mncl_dfn, mncl_dfan = [], [], [], []
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
                ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
                ncl_sts_path = vws_path / "nuclei_stats.pkl"
                asc_dst_path = vws_path / "asc_distances.pkl"
                ncl_df_path = vws_path / "nuclei_df.csv"
                ncl_dfa_path = vws_path / "nuclei_dfa.csv"
                ncl_dfn_path = vws_path / "nuclei_dfn.csv"
                ncl_dfan_path = vws_path / "nuclei_dfan.csv"
                asc_df_path = vws_path / "asc_df.csv"
                asc_dfn_path = vws_path / "asc_dfn.csv"
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
                
                asc_df, asc_dfn = format_asc_distances(
                    asc_dst, ncl_prd, self.classes, stk_level=self.stk_level,
                    bin_size=self.asc_bin_size, bin_max=self.asc_bin_max, 
                    )
                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - format data")
                                      
                # Plot data ---------------------------------------------------
                
                t0 = time.time()
                
                fig = plot(ncl_df, ncl_dfa, ncl_dfn, ncl_dfan, 
                    self.channels, self.classes, self.colors)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - Plot data")
                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                ncl_df.to_csv(ncl_df_path, index=False)
                ncl_dfa.to_csv(ncl_dfa_path, index=False)
                ncl_dfn.to_csv(ncl_dfn_path, index=False)
                ncl_dfan.to_csv(ncl_dfan_path, index=False)
                asc_df.to_csv(asc_df_path, index=False)
                asc_dfn.to_csv(asc_dfn_path, index=False)
                fig.savefig(ncl_plot_path, dpi=150, bbox_inches="tight")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                # Append
                mncl_df.append(ncl_df)
                mncl_dfa.append(ncl_dfa)
                mncl_dfn.append(ncl_dfn)
                mncl_dfan.append(ncl_dfan)
                aasc_df.append(asc_df)
                aasc_dfn.append(asc_dfn)
                
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
            aasc_df = pd.concat(aasc_df).groupby(level=0).mean()
            aasc_dfn = pd.concat(aasc_dfn).groupby(level=0).mean()
            
            # Plot ------------------------------------------------------------
            
            t0 = time.time()
            
            fig = plot(mncl_df, mncl_dfa, mncl_dfn, mncl_dfan, 
                self.channels, self.classes, self.colors)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - plot")
            
            # Save ------------------------------------------------------------
            
            t0 = time.time()
            
            mncl_df.to_csv(mncl_df_path, index=False)
            mncl_dfa.to_csv(mncl_dfa_path, index=False)
            mncl_dfn.to_csv(mncl_dfn_path, index=False)
            mncl_dfan.to_csv(mncl_dfan_path, index=False)
            aasc_df.to_csv(aasc_df_path, index=False)
            aasc_dfn.to_csv(aasc_dfn_path, index=False)
            fig.savefig(mncl_plot_path, dpi=150, bbox_inches="tight")
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    
#%%
    
    # # Path
    # vws_idx = 0
    # vws_path = main.vws_paths[vws_idx]
    # ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    # ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
    # asc_prc_path = vws_path / "asc_processed.tif"
    
    # # Load
    # with open(ncl_ctd_path, "rb") as file:
    #     ncl_ctd = pickle.load(file)
    # with open(ncl_prd_path, "rb") as file:
    #     ncl_prd = pickle.load(file)
    # asc_prc = io.imread(asc_prc_path)
    
    # asc_dst = get_asc_distances(
    #     asc_prc, ncl_ctd, main.classes,
    #     stk_level=main.stk_level
    #     )
    
    # asc_df, asc_dfn = format_asc_distances(
    #     asc_dst, ncl_prd, main.classes,
    #     bin_size=main.asc_bin_size, bin_max=main.asc_bin_max
    #     )
        
#%%

    # import matplotlib.pyplot as plt
        
    # # plot
    
    # gcolor = "0.5"
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    
    # for cl, cla in enumerate(main.classes):
    #     ax.plot(bins, counts_rel[:, cl], color=main.colors[cl], label=cla)
    
    # # Mean line
    # ax.axhline(y=1, color=gcolor, linestyle=":", linewidth=1)
        
    # # Legend
    # ax.set_ylabel("Ratio")
    # ax.set_xlabel("ASC cluster distance (pixels)")
    # ax.set_title("Nuclei class observed-to-expected ratio", pad=10)
    # ax.legend()
    
    
