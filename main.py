#%% Imports -------------------------------------------------------------------

import gc
import time
import pickle
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    prep_preview, norm_12_to_8bits, norm_data, 
    get_slide_metadata, load_view, get_preview_stats,
    get_nuclei_mask, get_nuclei_centroids, get_nuclei_stats,
    save_compressed_stack, load_compressed_stack,
    save_compressed_mask, load_compressed_mask,
    get_crops, prepare_crops,
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
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    "model_classes_name" : "model-cls_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # predict_nuclei()
    "thresh_0"  : 0.25,
    "thresh_1"  : 0.50,    
    
    # Measure()
    "crop_size" : 32,

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
                    t0 = time.time()
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
    
                    # Preview raw
                    pvw_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.pvw_level, roi_idx=roi_idx,
                        )
                    
                    # Preview mask
                    pvw_prp = prep_preview(pvw_raw, channels=[2, 5, 6])
                    pvw_msk = model_preview.predict(
                        pvw_prp, batch_size=32) > 0.5
                    pvw_msk = (pvw_msk * 255).astype("uint8")
                    
                    # Preview stats
                    pvw_sts = get_preview_stats(pvw_raw, pvw_msk, self.channels)
                    
                    # Preview normalization
                    pvw_nrm = (norm_data(pvw_raw, pvw_sts) * 255).astype("uint8")
                    
                    # Stack raw
                    stk_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.stk_level, roi_idx=roi_idx,
                        )
                    stk_raw = norm_12_to_8bits(stk_raw)
                    
                    # Stack normalization
                    stk_nrm = (norm_data(stk_raw, pvw_sts) * 255).astype("uint8")
                    
                    # Save
                    io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    save_compressed_stack(stk_raw, stk_raw_path)
                    save_compressed_stack(stk_nrm, stk_nrm_path)
                    with open(pvw_sts_path, "wb") as file:
                        pickle.dump(pvw_sts, file)     
                        
                    t1 = time.time()
                    print(f"duration : {t1 - t0:.3f}s")
                            
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
            
            if not ncl_ctd_path.exists():
                       
                # Log
                t0 = time.time()
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
                ncl = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=0, roi_idx=roi_idx,
                    )
                                   
                # Nuclei prediction
                ncl_prd = model_nuclei.predict(
                    norm_data(ncl, pvw_sts, channel="DAPI"), 
                    patch_overlap=32, batch_size=64, chunk_size=5000, 
                    )
    
                # Nuclei mask
                ncl_msk = get_nuclei_mask(
                    ncl_prd, thresh_0=self.thresh_0, thresh_1=self.thresh_1)
    
                # Nuclei centroids                    
                ncl_ctd = get_nuclei_centroids(
                    ncl_msk, n_levels=len(mtd["level"]))
                
                # Save
                save_compressed_mask(ncl_msk, ncl_msk_path)
                with open(ncl_ctd_path, "wb") as file:
                    pickle.dump(ncl_ctd, file)
    
                # Clear
                del ncl, ncl_prd, ncl_msk, ncl_ctd
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")
                                                         
#%% Class(Main) : predict_classes() -------------------------------------------
            
    def predict_classes(self):
        
        for p, vws_path in enumerate(self.vws_paths):
            
            # Paths
            stk_nrm_path = vws_path / "stack_norm.npz"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
            ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
            
            if not ncl_ent_path.exists():
            
                # Log
                t0 = time.time()
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
                    
                # Load
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                stk_nrm = load_compressed_stack(stk_nrm_path)
                    
                # Get crops
                crop_size = int(self.model_classes_name.split("_")[1])
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=crop_size, 
                    stk_level=self.stk_level
                    )
                
                # Predict nuclei classes
                crp_nrm = prepare_crops(crp_nrm)
                ncl_prd = model_classes.predict(crp_nrm)
                ncl_ent = entropy(ncl_prd, axis=1, base=len(self.classes))
                ncl_prd = np.argmax(ncl_prd, axis=1)
    
                # Save
                with open(ncl_prd_path, "wb") as file:
                    pickle.dump(ncl_prd, file)
                with open(ncl_ent_path, "wb") as file:
                    pickle.dump(ncl_ent, file)
                    
                # Clear
                del stk_nrm, ncl_ctd, crp_nrm, ncl_prd, ncl_ent
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")

#%% Class(Main) : measure() ---------------------------------------------------

    def measure(self):
                
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"     

            # if not ncl_sts_path.exists():           

            # Log
            t0 = time.time()
            log_str = f"Main() : measure() : {vws_path.name}"
            print("-" * len(log_str))
            print(log_str)
            print("-" * len(log_str))                           

            # Load
            with open(ncl_ctd_path, "rb") as file:
                ncl_ctd = pickle.load(file)
            stk_nrm = load_compressed_stack(stk_nrm_path)
            
            # Get crops
            crp_nrm = get_crops(
                stk_nrm, ncl_ctd, 
                crop_size=self.crop_size, 
                crop_fraction=None, 
                stk_level=self.stk_level,
                )
            crp_nrm = crp_nrm.astype("float32") / 255 # normalize
            
            # Get nuclei stats
            ncl_sts = get_nuclei_stats(crp_nrm, self.channels)
                
            # Save
            with open(ncl_sts_path, "wb") as file:
                pickle.dump(ncl_sts, file)
                
            t1 = time.time()
            print(f"duration : {t1 - t0:.3f}s")
                
#%% Class(Main) : analyse() ---------------------------------------------------  

    def analyse(self):
        
        for p, vws_path in enumerate(self.vws_paths):
            
            # Paths
            ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
            
            # Load
            with open(ncl_prd_path, "rb") as file:
                ncl_prd = pickle.load(file)
            with open(ncl_sts_path, "rb") as file:
                ncl_sts = pickle.load(file)
                
        pass              
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    
#%% 
    
    # Imports
    import pandas as pd

    # -------------------------------------------------------------------------

    def format_preview_results(pvw_sts, ):
        pass

    def format_nuclei_results(ncl_prd, ncl_sts, ):
        pass
    
    # -------------------------------------------------------------------------

    # Initialize
    vws_idx = 0
    vws_path = main.vws_paths[vws_idx]
    channels = main.channels
    classes = main.classes
    crop_size = main.crop_size
    stk_level = main.stk_level
    
    # -------------------------------------------------------------------------

    # Paths
    ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
    ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
    ncl_sts_path = vws_path / "nuclei_stats.pkl"
    
    # Load
    with open(ncl_prd_path, "rb") as file:
        ncl_prd = pickle.load(file)
    with open(ncl_ent_path, "rb") as file:
        ncl_ent = pickle.load(file)
    with open(ncl_sts_path, "rb") as file:
        ncl_sts = pickle.load(file)
        
    # -------------------------------------------------------------------------
    
    ncl_df = pd.DataFrame()
    n_nuclei = ncl_prd.shape[0]
    ncl_df["vws"    ] = [vws_path.name] * n_nuclei
    ncl_df["idx"    ] = np.arange(n_nuclei)
    ncl_df["class"  ] = np.array(classes)[ncl_prd]
    ncl_df["entropy"] = ncl_ent
    for msr in ["avg", "std"]:
        for chn in channels:
            key = f"{chn}_{msr}"
            ncl_df[key] = ncl_sts[msr][chn]
    ncl_df_avg = ncl_df.groupby("class").mean(numeric_only=True)

    
