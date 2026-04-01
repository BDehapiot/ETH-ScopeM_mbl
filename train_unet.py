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
    "run_patches_0"  : 0, # new
    "run_patches_1"  : 0, # from masks
    "run_annotate"   : 0,
    "run_train"      : 1,
    
    # Paths
    "train_path" : Path.cwd() / "data" / "train", 
    
    # extract_patches()
    "n_patches"             : 20, # save n_patches per DAPI image
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
    "patch_overlap"      : 0,
    "mask_method"        : "edt",
            
    # Augment
    "augment_iterations" : 1024,
    "augment_invert_p"   : 0,
    "augment_gamma_p"    : 0.5,
    "augment_gblur_p"    : 0.5,
    "augment_noise_p"    : 0.5,
    "augment_flip_p"     : 0.5,
    "augment_distord_p"  : 0.5,
    
    }

#%% Class(TrainUNet) --------------------------------------------------------------

class TrainUNet:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_patches_0:
            self.extract_patches_0()
        if self.run_patches_1:
            self.extract_patches_1()
        if self.run_annotate:
            self.annotate()
        if self.run_train:
            self.train()

#%% Class(Train) extract_patches() --------------------------------------------

    def extract_patches_0(self):
        
        dap_paths = [
            vws_path / "dapi_level-0.tif"
            for vws_path in self.main.vws_paths
            ]
        
        for p, path in enumerate(dap_paths):
            dapi = io.imread(path)
            patches = extract_patches(
                dapi, self.extract_patch_size, self.extract_patch_overlap)
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
                
    def extract_patches_1(self):
        
        msk_paths = list(self.train_path.glob("*_mask.tif"))
        for path in msk_paths:
            
            # Parse stem
            stem = path.stem
            vws_path = self.main.data_path / stem[:12] / stem[13:20]
            dap_path = vws_path / "dapi_level-0.tif"
            idx = int(stem.split("_")[-2])
            
            # Extract & save patch
            dapi = io.imread(dap_path)
            patch = extract_patches(
                dapi, self.extract_patch_size, self.extract_patch_overlap)[idx]
            patch_name = str(path).replace("_mask", "")
            io.imsave(
                self.train_path / patch_name,
                patch, check_contrast=False,
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
    train = TrainUNet(main, parameters=unet_parameters)