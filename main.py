#%% Imports -------------------------------------------------------------------

import pickle
from skimage import io
from pathlib import Path

# functions
from functions import (
    norm_12_to_8bits, norm_pct_8bits, norm_preview, norm_dapi,
    get_slide_metadata, load_view, get_statistics,
    )

# bdtools
from bdtools.unet.unet import UNet

# Skimage


#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model_256_binary_1024-56",
    
    # extract()
    "pvw_level" : 4,
    "dap_level" : 0,

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
                pvw_path = vws_path / "preview.tif"
                msk_path = vws_path / "mask.tif"
                sts_path = vws_path / "statistics.pkl"
                dap_path = vws_path / "dapi.tif"
                                
                if not vws_path.exists():
                    print(f"Main() : extract() : {vws_path.name}")
                    
                    vws_path.mkdir(parents=True, exist_ok=True) 
                    
                    # Model
                    if "unet_preview" not in locals():
                        unet_preview = UNet(
                            parameters=None, 
                            model_path=Path.cwd() / self.model_preview_name
                            )
                    
                    # Preview
                    pvw = load_view(
                        path, metadata=mtd, channels="all", 
                        level=self.pvw_level, view_idx=view_idx,
                        )
                    
                    # Mask
                    pvw_norm = norm_preview(
                        pvw, channels=[2, 5, 6], dtype="float32")
                    msk = norm_pct_8bits(unet_preview.predict(pvw_norm) > 0.5)
                    
                    # Statistics
                    sts = get_statistics(pvw, msk)
                    
                    # Dapi
                    dap = load_view(
                        path, metadata=mtd, channels="DAPI", 
                        level=self.dap_level, view_idx=view_idx,
                        )
                    dap = norm_dapi(
                        dap, msk, 
                        dap_level=self.dap_level, 
                        pvw_level=self.pvw_level, 
                        dtype="uint8"
                        )
                    
                    # Save
                    io.imsave(pvw_path, pvw, check_contrast=False)
                    io.imsave(msk_path, msk, check_contrast=False)
                    io.imsave(dap_path, dap, check_contrast=False)
                    with open(sts_path, "wb") as file:
                        pickle.dump(sts, file)        
                        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)