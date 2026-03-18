#%% Imports -------------------------------------------------------------------

import shutil
import numpy as np
from skimage import io
from slide import Slide

# bdtools
from bdtools.norm import norm_pct
from bdtools.patch import get_patches

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    }

#%% Class(Extract) ------------------------------------------------------------

class Extract:
    def __init__(self, data_path, parameters=None):
        self.data_path = data_path
        
        # Run
        self.initialize()
        
#%% Class(Extract) initialize() -----------------------------------------------

    def initialize(self):
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))

#%% Class(Extract) extract_patches() ------------------------------------------

#%% Class(Extract) extract() --------------------------------------------------

    def extract(self):
        
        # Setup output directory
        self.out_paths, self.slide_names = [], []
        for path in self.slide_paths:
            out_path = self.data_path / path.stem
            if out_path.exists():
                shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            self.out_paths.append(out_path)
            self.slide_names.append(path.stem[:12])
            
        
            
        # # Save preview (level = 4)
        # for p, path in enumerate(self.slide_paths):
        #     print(path.name)
        #     slide = Slide(path)
        #     metadata = slide.metadata
        #     for view in range(len(metadata["positions"])):
        #         if len(metadata["levels"]) > 4:
        #             preview = slide.load(channels="all", level=4, view=view)
        #             preview = np.mean(preview, axis=0)
        #             preview = (norm_pct(preview) * 255).astype("uint8")
        #             io.imsave(
        #                 self.out_paths[p] / f"preview_{view:02d}.tif", 
        #                 preview, check_contrast=False,
        #                 )
        
        # for p, path in enumerate(self.slide_paths):
        #     print(path.name)
        #     slide = Slide(path)
        #     metadata = slide.metadata
        #     for view in range(len(metadata["positions"])):
        #         arr = slide.load(channels="DAPI", level=1, view=view)
        #         arr = (norm_pct(arr) * 255).astype("uint8")
        #         slide_name = self.slide_names[p]
        #         save_name = (f"{slide_name}_{view:02d}_DAPI_level-1.tif")
        #         io.imsave(
        #             self.out_paths[p] / save_name, 
        #             arr, check_contrast=False,
        #             )
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time
    from pathlib import Path
    
    # -------------------------------------------------------------------------
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    
    # -------------------------------------------------------------------------
    
    extract = Extract(data_path, parameters=parameters)
    # paths = extract.paths