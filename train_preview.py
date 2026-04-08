#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# bdtools
from bdtools.unet.unet import UNet
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

unet_parameters = {
    
    # Procedure
    "run_patches_0" : 0, # new
    "run_annotate"  : 0,
    "run_train"     : 1,
    
    # Paths
    "train_path" : Path.cwd() / "data" / "train_preview", 
    
    # extract_patches()
    "n_patches"             : 10, # save n_patches per DAPI image
    "extract_patch_size"    : 256,
    "extract_patch_overlap" : 128,
    
    # train() -----------------------------------------------------------------
    
    # Paths
    "root_path"          : None,
    "model_name"         : None,

    # Build
    "input_shape"        : (None, None, 1),
    "backbone"           : "resnet18",
    "activation"         : "sigmoid",
        
    # Train
    "epochs"             : 256,
    "batch_size"         : 64,
    "validation_split"   : 0.2,
    "metric"             : "soft_dice_coef",
    "learning_rate"      : 0.001,
    "patience"           : 64,

    # Prepare
    "patch_size"         : 256,
    "patch_overlap"      : 128,
    "mask_method"        : "binary",
            
    # Augment
    "augment_iterations" : 1024,
    "augment_invert_p"   : 0,
    "augment_gamma_p"    : 0.5,
    "augment_gblur_p"    : 0.5,
    "augment_noise_p"    : 0.5,
    "augment_flip_p"     : 0.5,
    "augment_distord_p"  : 0.5,
    
    }

#%% Class(TrainPreview) --------------------------------------------------------------

class TrainPreview:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_patches_0:
            self.extract_patches_0()
        if self.run_annotate:
            self.annotate()
        if self.run_train:
            self.train()

#%% Class(Train) extract_patches() --------------------------------------------

    def extract_patches_0(self):
        
        pvw_paths = self.main.data_path.rglob("preview*")
        
        for p, path in enumerate(pvw_paths):
            pvw = io.imread(path)
            patches = extract_patches(
                pvw, self.extract_patch_size, self.extract_patch_overlap)
            idxs =  np.random.choice(
                np.arange(0, len(patches)), size=self.n_patches, replace=False)
            for idx in idxs:
                patch_name = (
                    f"{str(path.parent.parent.name)}_"
                    f"{str(path.parent.name)}_"
                    f"{path.stem}_"
                    f"{idx:05d}.tif"
                    )
                io.imsave(
                    self.train_path / patch_name,
                    patches[idx], check_contrast=False,
                    )
                
#%% Class(Train) annotate() ---------------------------------------------------

    def annotate(self):
        Annotate(self.train_path)  
        
#%% Class(Train) train() ------------------------------------------------------

    def train(self):
        
        # Load data
        msk_paths = list(self.train_path.glob("*_mask.tif"))
        msks, imgs = [], []
        for path in msk_paths:
            msks.append(io.imread(path))
            imgs.append(io.imread(
                str(path).replace("_mask", "")
                ))
        msks = np.stack(msks)
        imgs = np.stack(imgs)
            
        # Normalize data (images)
        imgs = imgs.astype("float32") / 255
            
        # Setup model
        if self.model_name is not None:
            model_path = Path.cwd() / self.model_name
            self.unet = UNet(parameters=None, model_path=model_path)
        else:
            self.unet = UNet(parameters=self.parameters, model_path=None)
        
        # Train model
        self.unet.train(imgs, msks)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainPreview(main, parameters=unet_parameters)