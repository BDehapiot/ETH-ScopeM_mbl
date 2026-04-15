#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# function(s)
from functions import load_view, norm_data

# bdtools
from bdtools.unet.unet import UNet
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

unet_parameters = {
    
    # Procedure
    "run_patches_0" : 0, # new
    "run_patches_1" : 1, # from preexisting masks
    "run_annotate"  : 0,
    "run_train"     : 0,
    
    # Paths
    "train_path" : Path.cwd() / "data" / "train_dapi", 
    
    # extract_patches()
    "n_patches"             : 10, # save n_patches per preview image
    "extract_patch_size"    : 256,
    "extract_patch_overlap" : 128,
    
    # train() -----------------------------------------------------------------
    
    # Paths
    "root_path"          : None,
    "model_name"         : "model_256_edt_1024-48",

    # Build
    "input_shape"        : (None, None, 1),
    "backbone"           : "resnet18",
    "activation"         : "sigmoid",
        
    # Train
    "epochs"             : 256,
    "batch_size"         : 64,
    "validation_split"   : 0.2,
    "metric"             : "soft_dice_coef",
    "learning_rate"      : 0.0001,
    "patience"           : 64,

    # Prepare
    "patch_size"         : 256,
    "patch_overlap"      : 128,
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

#%% Class(TrainDapi) ----------------------------------------------------------

class TrainDapi:
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
        
        """
        to be implemented
        """
        
        pass
                
    def extract_patches_1(self):
        
        msk_paths = list(self.train_path.glob("*_mask.tif"))
        
        for path in msk_paths:
            
            # Parse paths
            qpt_path  = self.data_path / (path.stem[:26] + ".qptiff")
            mtd_path  = self.data_path / (path.stem[:26] + "_metadata.pkl")
            its_path  = self.data_path / path.stem[:29] / "intensities.pkl"
            view_idx  = int(path.stem[27:29])
            patch_idx = int(path.stem.split("_")[-2])
                        
            # Load
            with open(mtd_path, "rb") as file:
                mtd = pickle.load(file)
            with open(its_path, "rb") as file:
                its = pickle.load(file)
            dap = load_view(
                qpt_path, metadata=mtd, channels="DAPI", 
                level=0, view_idx=view_idx,
                )
            dap = norm_data(dap, its, channel_idx=0)
            
            # Extract patch
            patch = extract_patches(
                dap, self.extract_patch_size, self.extract_patch_overlap)[patch_idx]
            
            # Save
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
    train = TrainDapi(main, parameters=unet_parameters)
    
    