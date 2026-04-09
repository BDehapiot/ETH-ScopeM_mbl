#%% Imports -------------------------------------------------------------------

import pickle
from skimage import io
from pathlib import Path

# functions
from functions import (
    norm_12_to_8bits, norm_pct_8bits,
    get_slide_metadata, load_view
    )

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    
    # Preview
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
                
        self.mtd_paths, self.vws_paths = [], []
        for p, path in enumerate(self.qpt_paths): 
            
            # Metadata
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
            for view_idx in range(len(mtd["roi"])):
                vws_name = str(path.name).replace(".qptiff", f"_view_{view_idx:02d}")
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
                        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)