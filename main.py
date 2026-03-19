#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import load_metadata, load_arr

# bdtools
from bdtools.norm import norm_pct

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    
    # initialize()
    "preview_level" : 4,
        
    }

#%% Class(Main) ------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.initialize()
        
#%% Class(Main) : initialize() ---------------------------------------------

    def initialize(self):
        
        # Paths
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.output_paths = [
            self.data_path / f"{path.stem[:12]}"
            for path in self.slide_paths
            ]

        # load/save metadata & preview
        self.metadata_paths = []
        self.preview_paths = []
        self.dapi_paths = []
        for p, path in enumerate(self.slide_paths): 
            
            print(f"initialize() : {path.name}")
            
            output_path = self.output_paths[p]
            metadata_path = output_path / "metadata.pkl"
            preview_path = output_path / "preview.tif"
            self.metadata_paths.append(metadata_path)
            self.preview_paths.append(preview_path)
            
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True) 
            
            if not metadata_path.exists():
                metadata = load_metadata(path)
                with open(metadata_path, "wb") as file:
                    pickle.dump(metadata, file)
            else:
                with open(metadata_path, "rb") as file:
                    metadata = pickle.load(file)
                
            if not preview_path.exists():
                preview = load_arr(
                    path, metadata=metadata, 
                    channels="all", level=self.preview_level
                    )
                preview = np.mean(preview, axis=0)
                preview = (norm_pct(preview) * 255).astype("uint8")
                io.imsave(preview_path, preview, check_contrast=False)
                
            for idx in range(len(metadata["roi"])):
                dapi_path = output_path / f"dapi_{idx:02d}.tif"
                self.dapi_paths.append(dapi_path)
                if not dapi_path.exists():
                    dapi = load_arr(
                        path, metadata=metadata, 
                        channels="DAPI", level=0, roi_idx=idx,
                        )
                    dapi = (norm_pct(dapi) * 255).astype("uint8")
                    io.imsave(dapi_path, dapi, check_contrast=False)
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=parameters)
    