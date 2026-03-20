#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# bdtools
from bdtools.patch import get_patches

#%% Inputs --------------------------------------------------------------------

train_parameters = {
    
    # Paths
    "data_path"  : Path("D:\local_mbl\data"),
    "train_path" : Path.cwd() / "data" / "train", 
    
    # extract_patches()
    "n_patches"     : 20, # save n_patches per DAPI image
    "patch_size"    : 512,
    "patch_overlap" : 256,
    
    }

#%% Class(Train) --------------------------------------------------------------

class Train:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.extract_patches()

#%% Class(Train) initialize() -------------------------------------------------
                
#%% Class(Train) extract_patches() --------------------------------------------

    def extract_patches(self):
        
        for p, path in enumerate(self.main.dapi_paths):
            if p == 0:
                dapi = io.imread(path)
                patches = get_patches(
                    dapi, self.patch_size, self.patch_overlap)
                idxs =  np.random.choice(
                    np.arange(0, len(patches)), size=self.n_patches, replace=False)
                for idx in idxs:
                    patch_name = (
                        f"{str(path.parent.name)}_"
                        f"{path.stem}_"
                        f"{idx:04d}.tif"
                        )
                    io.imsave(
                        self.train_path / patch_name,
                        patches[idx], check_contrast=False,
                        )
        
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = Train(main, parameters=train_parameters)
    # patches = train.patches
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(np.stack(patches))
    