#%% Imports -------------------------------------------------------------------

import gc
import pickle
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    prep_preview, norm_data,
    get_slide_metadata, load_view, get_intensities,
    get_nuclei_mask, get_nuclei_centroids,
    save_bin_mask, load_bin_mask,
    )

# bdtools
from bdtools.unet.unet import UNet

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model_256_binary_1024-56",
    "model_nuclei_name"  : "model_256_edt_1024-48",
    
    # extract()
    "pvw_level" : 4,
    
    # predict()
    "thresh_0"  : 0.25,
    "thresh_1"  : 0.50,

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
        self.extract()
        self.predict()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        self.qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):

        self.mtd_paths, self.vws_paths = [], []
        for p, path in enumerate(self.qpt_paths): 
            
            # Metadata --------------------------------------------------------
            
            # Paths
            mtd_name = str(path.name).replace(".qptiff", "_metadata.pkl")
            mtd_path = self.data_path / mtd_name
            self.mtd_paths.append(mtd_path)
            
            if not mtd_path.exists():
                mtd = get_slide_metadata(path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                    
            # Data ------------------------------------------------------------
                    
            for view_idx in range(len(mtd["roi"])):
                
                # Paths
                vws_name = str(path.name).replace(".qptiff", f"_{view_idx:02d}")
                vws_path = self.data_path / vws_name
                self.vws_paths.append(vws_path)
                pvw_raw_path = vws_path / "preview_raw.tif"
                pvw_msk_path = vws_path / "preview_mask.tif"
                pvw_nrm_path = vws_path / "preview_norm.tif"
                its_path = vws_path / "intensities.pkl"

                if not vws_path.exists():
                    
                    # Model
                    if "unet_preview" not in locals():
                        unet_preview = UNet(
                            parameters=None, 
                            model_path=Path.cwd() / self.model_preview_name
                            )
                    
                    print(f"Main() : extract() : {vws_path.name}")
                    
                    vws_path.mkdir(parents=True, exist_ok=True) 

                    # Preview raw
                    pvw_raw = load_view(
                        path, metadata=mtd, channels="all", 
                        level=self.pvw_level, view_idx=view_idx,
                        )
                    
                    # Preview mask
                    pvw_prp = prep_preview(pvw_raw, channels=[2, 5, 6])
                    pvw_msk = unet_preview.predict(pvw_prp) > 0.5
                    pvw_msk = (pvw_msk * 255).astype("uint8")
                    
                    # Intensities
                    its = get_intensities(pvw_raw, pvw_msk)
                    
                    # Preview norm.
                    pvw_nrm = (norm_data(pvw_raw, its) * 255).astype("uint8")
                                        
                    # Save
                    io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    with open(its_path, "wb") as file:
                        pickle.dump(its, file)        
                        
#%% Class(Main) : predict() ---------------------------------------------------
                   
    def predict(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            qpt_path = self.data_path / (vws_path.stem[:-3] + ".qptiff")
            mtd_path = self.data_path / (vws_path.stem[:-3] + "_metadata.pkl")
            its_path = vws_path / "intensities.pkl"
            view_idx = int(vws_path.stem[-2:])
            ncl_msk_path = vws_path / "nuclei_mask.npz"
            ncl_ctd_path = vws_path / "centroids.pkl"
            
            if not ncl_msk_path.exists():
                
                # Model
                if "unet_nuclei" not in locals():
                    unet_nuclei = UNet(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_nuclei_name
                        )
                
                print(f"Main() : predict() : {vws_path.name}")
                
                # Load
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(its_path, "rb") as file:
                    its = pickle.load(file)
                ncl = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=0, view_idx=view_idx,
                    )
                                   
                # Nuclei prediction
                ncl_prd = unet_nuclei.predict(
                    norm_data(ncl, its, channel_idx=0), 
                    batch_size=64, chunk_size=5000, patch_overlap=32,
                    )

                # Nuclei mask
                ncl_msk = get_nuclei_mask(
                    ncl_prd, thresh_0=self.thresh_0, thresh_1=self.thresh_1)

                # Nuclei centroids                    
                ncl_ctd = get_nuclei_centroids(
                    ncl_msk, n_levels=len(mtd["level"]))

                # Save
                save_bin_mask(ncl_msk, ncl_msk_path)
                with open(ncl_ctd_path, "wb") as file:
                    pickle.dump(ncl_ctd, file)
                                    
                # Clear
                del ncl, ncl_prd, ncl_msk
                tf.keras.backend.clear_session()
                gc.collect()
                
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):
        
        
        
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    