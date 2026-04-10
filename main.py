#%% Imports -------------------------------------------------------------------

import pickle
from skimage import io
from pathlib import Path

# functions
from functions import (
    norm_12_to_8bits, norm_pct_8bits, prep_preview,
    get_slide_metadata, load_view,
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
    
    # extract()
    "pvw_level" : 4,

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
           
        # Metadata
        self.mtd_paths = []
        for p, path in enumerate(self.qpt_paths): 

            mtd_name = str(path.name).replace(".qptiff", "_metadata.pkl")
            mtd_path = self.data_path / mtd_name
            self.mtd_paths.append(mtd_path)
            
            if not mtd_path.exists():
                print(f"Main() : metadata : {path.name}")
                mtd = get_slide_metadata(path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                    
        # Previews
        self.vws_paths = []
        for p, path in enumerate(self.qpt_paths): 
            
            with open(self.mtd_paths[p], "rb") as file:
                mtd = pickle.load(file)
            
            for view_idx in range(len(mtd["roi"])):
                
                vws_name = str(path.name).replace(".qptiff", f"_{view_idx:02d}")
                vws_path = self.data_path / vws_name
                self.vws_paths.append(vws_path)
                
                if not vws_path.exists():
                    print(f"Main() : preview : {vws_path.name}")
                    vws_path.mkdir(parents=True, exist_ok=True) 
                    pvw = load_view(
                        path, metadata=mtd, channels="all", 
                        level=self.pvw_level, view_idx=view_idx,
                        )
                    io.imsave(
                        vws_path / "preview.tif", pvw, check_contrast=False)
                    
        # Masks
        self.msk_paths = []
        for p, path in enumerate(self.vws_paths): 
                            
            msk_path = path / "mask.tif"
            self.msk_paths.append(msk_path) 
                        
            if not msk_path.exists():
                print(f"Main() : mask : {path.name}")
                
                if "unet_preview" not in locals():
                    unet_preview = UNet(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_preview_name
                        )
                
                pvw = prep_preview(io.imread(path / "preview.tif"))
                pvw = pvw.astype("float32") / 255
                msk = norm_pct_8bits(unet_preview.predict(pvw) > 0.5)
                io.imsave(
                    path / "mask.tif", msk, check_contrast=False)
                        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)