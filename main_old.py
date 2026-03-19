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
        
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))
        
        # Load/save metadata
        self.output_paths, self.metadata_paths  = [], []
        for path in self.slide_paths:  
            output_path = self.data_path / f"{path.stem[:12]}"
            metadata_path = output_path / "metadata.pkl"
            self.output_paths.append(output_path)
            self.metadata_paths.append(metadata_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            if not metadata_path.exists(): 
                print(f"Load metadata : {path.name}")
                metadata = load_metadata(path)
                with open(metadata_path, "wb") as file:
                    pickle.dump(metadata, file)
            
        # Load/save preview
        self.preview_paths = []
        for p, path in enumerate(self.slide_paths): 
            with open(self.metadata_paths[p], "rb") as file:
                metadata = pickle.load(file) 
            for view in range(len(metadata["positions"])):
                preview_path = self.output_paths[p] / f"view-{view:02d}_preview.tif"
                self.preview_paths.append(preview_path)
                if not preview_path.exists():
                    preview = load_arr(
                        path, metadata=metadata, channels="all", view=view,
                        level=self.preview_level,
                        )
                    preview = np.mean(preview, axis=0)
                    preview = (norm_pct(preview) * 255).astype("uint8")
                    io.imsave(preview_path, preview, check_contrast=False)
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=parameters)
    