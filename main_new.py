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
    
    # Maps
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
    
    # Paths
    "data_path" : Path("D:\local_mbl\data_test"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    "model_classes_name" : "model-cls_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # # predict_nuclei()
    # "thresh_0"  : 0.25,
    # "thresh_1"  : 0.50,    
    
    # # Measure()
    # "crop_size" : 32,

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
        # self.predict_nuclei()
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
                stk_raw_path = vws_path / "stack_raw.npz"
                stk_nrm_path = vws_path / "stack_norm.npz"

                if not stk_nrm_path.exists():
                    
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
                    
                    # Preview -------------------------------------------------
                    
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
                    
                    # Preview normalization
                    pvw_nrm = (norm_data(pvw_raw, pvw_sts) * 255).astype("uint8")
                    
                    t1 = time.time()
                    print(f"Get preview : {t1 - t0:.3f}s")
                    
                    # Stack ---------------------------------------------------
                    
                    # Stack raw
                    stk_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.stk_level, roi_idx=roi_idx,
                        )
                    stk_raw = norm_12_to_8bits(stk_raw)
                    
                    t0 = time.time()
                    
                    # Stack normalization
                    stk_nrm = (norm_data(stk_raw, pvw_sts) * 255).astype("uint8")
                    
                    t1 = time.time()
                    print(f"Get stack : {t1 - t0:.3f}s")
                                        
                    # # Save
                    # io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    # io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    # io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    # save_compressed_stack(stk_raw, stk_raw_path)
                    # save_compressed_stack(stk_nrm, stk_nrm_path)
                    # with open(pvw_sts_path, "wb") as file:
                    #     pickle.dump(pvw_sts, file)     
                        
                    t1f = time.time()
                    print(f"Total : {t1f - t0i:.3f}s")
                            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    
#%% Load ----------------------------------------------------------------------
    
    # Paths
    vws_idx = 0
    vws_path = main.vws_paths[vws_idx]    
    
    