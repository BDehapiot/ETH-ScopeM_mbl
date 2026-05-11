#%% Imports -------------------------------------------------------------------

import gc
import pickle
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    prep_preview, norm_12_to_8bits, norm_data, 
    get_slide_metadata, load_view, get_intensities,
    get_nuclei_mask, get_nuclei_centroids,
    save_compressed_stack, load_compressed_stack,
    save_compressed_mask, load_compressed_mask,
    get_crops,
    )

# bdtools
from bdtools.model.model import Model

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels"  : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # predict()
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
        self.predict()
        # self.measure()
        
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
                pvw_its_path = vws_path / "preview_intensities.pkl"
                stk_raw_path = vws_path / "stack_raw.npz"
                stk_nrm_path = vws_path / "stack_norm.npz"

                if not vws_path.exists():
                    
                    # Model
                    if "model_preview" not in locals():
                        model_preview = Model(
                            parameters=None, 
                            model_path=Path.cwd() / self.model_preview_name
                            )
                    
                    print(f"Main() : extract() : {vws_path.name}")
                    
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
                    
                    # Intensities
                    pvw_its = get_intensities(pvw_raw, pvw_msk)
                    
                    # Preview norm.
                    pvw_nrm = (norm_data(pvw_raw, pvw_its) * 255).astype("uint8")
                    
                    # Stack raw
                    stk_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.stk_level, roi_idx=roi_idx,
                        )
                    stk_raw = norm_12_to_8bits(stk_raw)
                    
                    # Stack norm
                    stk_nrm = (norm_data(stk_raw, pvw_its) * 255).astype("uint8")
                    
                    # Save
                    io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    save_compressed_stack(stk_raw, stk_raw_path)
                    save_compressed_stack(stk_nrm, stk_nrm_path)
                    with open(pvw_its_path, "wb") as file:
                        pickle.dump(pvw_its, file)        
                            
#%% Class(Main) : predict() ---------------------------------------------------
                   
    def predict(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            qpt_path = self.qpt_paths[p]
            mtd_path = self.mtd_paths[p]
            pvw_its_path = vws_path / "preview_intensities.pkl"
            ncl_msk_path = vws_path / "nuclei_mask.npz"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            roi_idx = int(vws_path.stem[-2:])
            
            if not ncl_msk_path.exists():
                
                # Model
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_nuclei_name
                        )
                
                print(f"Main() : predict() : {vws_path.name}")
                
                # Load
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_its_path, "rb") as file:
                    pvw_its = pickle.load(file)
                ncl = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=0, roi_idx=roi_idx,
                    )
                                   
                # Nuclei prediction
                ncl_prd = model_nuclei.predict(
                    norm_data(ncl, pvw_its, channel_idx=0), 
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
                del ncl, ncl_prd, ncl_msk
                tf.keras.backend.clear_session()
                gc.collect()
                
#%% Class(Main) : measure() ---------------------------------------------------

    def measure(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"            

            if p == 0:            

                print(f"Main() : measure() : {vws_path.name}")                

                # Load
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                stk_nrm = load_compressed_stack(stk_nrm_path)
                
                # Get crops
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=self.crop_size, 
                    crop_fraction=1, 
                    stk_level=self.stk_level,
                    )
                self.crp_nrm = crp_nrm
                

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    
#%% 

    # Imports
    import numpy as np
    from scipy import signal

    # Parameters
    channel = 0
    
    # Initialize
    crp_nrm = main.crp_nrm
    crp_nrm = crp_nrm.astype("float32") / 255
    
    # Create Gaussian kernel
    kernel = signal.windows.gaussian(32, std=8)
    kernel = np.outer(kernel, kernel)
    kernel /= np.sum(kernel) 
    kernel = kernel[None, None, :, :]
    
    # Apply Gaussian kernel
    crp_nrm *= kernel
    
    # Measure
    measures = {
        "avg" : [], "std" : [], "med" : [], "min" : [], "max" : []}
    for channel in range(crp_nrm.shape[1]):
        measures["avg"].append(np.mean(
            crp_nrm[:, channel, ...], axis=(1, 2)))
        measures["std"].append(np.std(
            crp_nrm[:, channel, ...], axis=(1, 2)))
        measures["med"].append(np.median(
            crp_nrm[:, channel, ...], axis=(1, 2)))
        measures["min"].append(np.min(
            crp_nrm[:, channel, ...], axis=(1, 2)))
        measures["max"].append(np.max(
            crp_nrm[:, channel, ...], axis=(1, 2)))
        
    # Plot
    