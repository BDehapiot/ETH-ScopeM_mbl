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
    "force_predict_nuclei"  : 1,
    "force_predict_classes" : 0,
    "force_measure"         : 0,
    "force_analyse"         : 0,
    
    # Paths
    "data_path" : Path("D:\local_mbl\data_test"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    "model_classes_name" : "model-cls_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # predict_nuclei()
    "thresh_0"  : 0.25,
    "thresh_1"  : 0.50,    
    
    # # Measure()
    # "crop_size" : 32,
    
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
        # self.predict_classes()
        # self.measure()
        # self.analyse()
        
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
            qpt_path = self.qpt_paths[p]
            mtd_path = self.mtd_paths[p]
            pvw_sts_path = vws_path / "preview_stats.pkl"
            ncl_msk_path = vws_path / "nuclei_mask.npz"
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
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_nuclei_name
                        )
                
                # Load
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                ncl = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=0, roi_idx=roi_idx,
                    )
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                
                # Predict -----------------------------------------------------
                
                # t0 = time.time()
                
                # ncl_prd = model_nuclei.predict(
                #     norm_data(ncl, pvw_sts, channel="DAPI"), 
                #     patch_overlap=32, batch_size=64, chunk_size=5000, 
                #     )
                
                # t1 = time.time()
                # print(f"{t1 - t0:.3f}s - predict")
    
                # Get mask ----------------------------------------------------
    
                # t0 = time.time()
    
                # ncl_msk = get_nuclei_mask(
                #     ncl_prd, thresh_0=self.thresh_0, thresh_1=self.thresh_1)
                
                # t1 = time.time()
                # print(f"{t1 - t0:.3f}s - get mask")
    
                # Get centroids -----------------------------------------------
    
                # t0 = time.time()
                               
                # ncl_ctd = get_nuclei_centroids(
                #     ncl_msk, n_levels=len(mtd["level"]))
                
                # t1 = time.time()
                # print(f"{t1 - t0:.3f}s - get centroids")
                
                # Save --------------------------------------------------------
                
                # t0 = time.time()
                
                # save_compressed_mask(ncl_msk, ncl_msk_path)
                # with open(ncl_ctd_path, "wb") as file:
                #     pickle.dump(ncl_ctd, file)
    
                # # Clear
                # del ncl, ncl_prd, ncl_msk, ncl_ctd
                # tf.keras.backend.clear_session()
                # gc.collect()
                
                # t1 = time.time()
                # print(f"{t1 - t0:.3f}s - save")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    