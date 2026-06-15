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
    prep_preview, norm_12_to_8bits, norm_data, 
    get_slide_metadata, load_view, 
    get_nuclei_mask, get_nuclei_centroids, 
    save_compressed_stack, load_compressed_stack,
    save_compressed_mask, load_compressed_mask,
    get_crops, prepare_crops,
    get_preview_stats, format_preview_stats,
    get_nuclei_stats, format_nuclei_stats,
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
    "force_predict_classes" : 0,
    "force_measure"         : 0,
    "force_analyse"         : 0,
    
    # Paths
    "data_path" : Path("D:\local_mbl\data_test"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name0" : "model-sm_nuclei_256_edt_192-1024",
    "model_nuclei_name1" : "model-sm_nuclei_128_edt_192-1024",
    "model_classes_name" : "model-cls_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # predict_nuclei()
    "ncl_level" : 0,
    "thresh_0"  : 0.25,
    "thresh_1"  : 0.50,
    
    # Measure()
    "crop_size" : 32,
    
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

                if not stk_nrm_path.exists() or self.force_extract:
                    
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
                    pvw_prp = prep_preview(pvw_raw, channels=[2, 5, 6])
                    pvw_msk = model_preview.predict(pvw_prp, batch_size=32) > 0.5
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
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            roi_idx = int(vws_path.stem[-2:])
                  
            if not ncl_ctd_path.exists() or self.force_predict_nuclei:
                       
                # Log
                t0i = time.time()
                log_str = f"Main() : predict_nuclei() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                model_nuclei_name = f"model_nuclei_name{self.ncl_level}"
                model_nuclei_path = Path.cwd() / getattr(self, model_nuclei_name)
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, 
                        model_path=model_nuclei_path
                        )
                                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                if self.ncl_level == self.stk_level:
                    ncl_nrm = io.imread(stk_nrm_path)[0]
                    ncl_nrm = ncl_nrm.astype("float32") / 255
                else: 
                    ncl_nrm = load_view(
                        qpt_path, metadata=mtd, channels="DAPI", 
                        level=self.ncl_level, roi_idx=roi_idx,
                        )
                    ncl_nrm = norm_data(ncl_nrm, pvw_sts, channel="DAPI")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Predict -----------------------------------------------------
                
                t0 = time.time()
                                
                if self.ncl_level == self.stk_level:
                    ncl_prd = model_nuclei.predict(
                        ncl_nrm, patch_overlap=16,
                        )
                else: 
                    ncl_prd = model_nuclei.predict(
                        ncl_nrm, patch_overlap=32, 
                        batch_size=64, chunk_size=5000, 
                        )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - predict")
    
                # Get mask ----------------------------------------------------
    
                t0 = time.time()
    
                ncl_msk = get_nuclei_mask(
                    ncl_prd, 
                    thresh_0=self.thresh_0, 
                    thresh_1=self.thresh_1, 
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
                del ncl_nrm, ncl_prd, ncl_msk, ncl_ctd
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
            
            if not ncl_ent_path.exists() or self.force_predict_classes:
            
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

                crp_nrm = prepare_crops(crp_nrm)
                ncl_prd = model_classes.predict(crp_nrm)
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
                del stk_nrm, ncl_ctd, crp_nrm, ncl_prd, ncl_ent
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
            ncl_sts_path = vws_path / "nuclei_stats.pkl"

            if not ncl_sts_path.exists() or self.force_measure:           

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
                stk_nrm = stk_nrm = io.imread(stk_nrm_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=self.crop_size, 
                    crop_fraction=None, 
                    stk_level=self.stk_level,
                    )
                crp_nrm = crp_nrm.astype("float32") / 255 # normalize
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                # Get stats ---------------------------------------------------
                    
                t0 = time.time()

                ncl_sts = get_nuclei_stats(crp_nrm, self.channels)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get stats")
                    
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_sts_path, "wb") as file:
                    pickle.dump(ncl_sts, file)
                    
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
        mncl_plot_path = self.data_path / "merged_nuclei_plot.png"
        
        if not mncl_dfan_path.exists() or self.force_analyse: 
        
            mncl_df, mncl_dfa, mncl_dfn, mncl_dfan = [], [], [], [],
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
                ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
                ncl_sts_path = vws_path / "nuclei_stats.pkl"
                ncl_df_path = vws_path / "nuclei_df.csv"
                ncl_dfa_path = vws_path / "nuclei_dfa.csv"
                ncl_dfn_path = vws_path / "nuclei_dfn.csv"
                ncl_dfan_path = vws_path / "nuclei_dfan.csv"
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
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                                    
                # Plot --------------------------------------------------------
                
                t0 = time.time()
                
                ncl_df, ncl_dfa, ncl_dfn, ncl_dfan = format_nuclei_stats(
                    ncl_prd, ncl_ent, ncl_sts, vws_path, 
                    self.channels, self.classes
                    )
                
                fig = plot(ncl_df, ncl_dfa, ncl_dfn, ncl_dfan, 
                    self.channels, self.classes, self.colors)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - plot")
                                                
                # Save --------------------------------------------------------
                
                t0 = time.time()
                
                ncl_df.to_csv(ncl_df_path, index=False)
                ncl_dfa.to_csv(ncl_dfa_path, index=False)
                ncl_dfn.to_csv(ncl_dfn_path, index=False)
                ncl_dfan.to_csv(ncl_dfan_path, index=False)
                fig.savefig(ncl_plot_path, dpi=150, bbox_inches="tight")
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - save")
                
                # Append
                mncl_df.append(ncl_df)
                mncl_dfa.append(ncl_dfa)
                mncl_dfn.append(ncl_dfn)
                mncl_dfan.append(ncl_dfan)
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                    
            # Merge -----------------------------------------------------------
            
            # Log
            t0i = time.time()
            log_str = "Main() : analyse : merge"
            print("-" * len(log_str))
            print(log_str)
            print("-" * len(log_str)) 
            
            # Concatenate            
            mncl_df = pd.concat(mncl_df, axis=0, ignore_index=True)
            mncl_dfa = pd.concat(mncl_dfa, axis=0, ignore_index=True)
            mncl_dfn = pd.concat(mncl_dfn, axis=0, ignore_index=True)
            mncl_dfan = pd.concat(mncl_dfan, axis=0, ignore_index=True)
            
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
            fig.savefig(mncl_plot_path, dpi=150, bbox_inches="tight")
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s - save")
            
            t1f = time.time()
            print(f"> {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    