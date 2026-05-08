#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import prep_preview

# bdtools
from bdtools.model.model import Model
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_patches"  : 0, # new
    "run_annotate" : 0,
    "run_train"    : 1,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_preview", 
    
    # Extract -----------------------------------------------------------------
    
    "n_patches"             : 10, # save n_patches per preview image
    "extract_patch_size"    : 256,
    "extract_patch_overlap" : 128,
    
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "sm",
    "input_shape"        : (None, None, 1),
    "backbone"           : "resnet18",
    "activation"         : "sigmoid",
    "loss"               : "bce",
    "metric"             : "mae",
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : 256,
    "patch_overlap"      : 128,
    "mask_method"        : "binary",

    # Train -------------------------------------------------------------------
    
    "display"            : 0,
    "epochs"             : 512,
    "batch_size"         : 32,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.0001,
    "patience"           : 128,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : 1024,
    "augment_gamma_p"    : 0.0,
    "augment_gblur_p"    : 0.0,
    "augment_noise_p"    : 0.0,
    "augment_flip_p"     : 0.5,
    "augment_distort_p"  : 0.5,
    
    "augment_parameters" : {
        
    # Gamma
    "gamma_low"          : 0.75,
    "gamma_high"         : 1.25,
    "gamma_chn"          : "independent",
    
    # Gaussian blur
    "gblur_sigma_low"    : 1,
    "gblur_sigma_high"   : 3,
    "gblur_chn"          : "shared",
    
    # Noise
    "noise_gain_low"     : 30,
    "noise_gain_high"    : 60,
    "noise_std_low"      : 3,
    "noise_std_high"     : 6,
    "noise_chn"          : "independent",
    
    # Grid distort
    "distort_steps_low"  : 1,
    "distort_steps_high" : 10,
    "distort_limit_low"  : 0.1,
    "distort_limit_high" : 0.5,
        
    },

    }

#%% Class(TrainPreview) --------------------------------------------------------------

class TrainPreview:
    def __init__(self, main, parameters=None, model_parameters=None):
        self.main = main
        self.parameters = parameters
        self.model_parameters = model_parameters
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_patches:
            self.extract_patches()
        if self.run_annotate:
            self.annotate()
        if self.run_train:
            self.train()

#%% Class(Train) extract_patches() --------------------------------------------

    def extract_patches(self):
        
        pvw_paths = self.main.data_path.rglob("preview_raw.tif")
        
        for p, path in enumerate(pvw_paths):
            
            pvw = io.imread(path)
            pvw_prp = prep_preview(pvw, channels=[2, 5, 6])
            pvw_prp = (pvw_prp * 255).astype("uint8")
            
            patches = extract_patches(
                pvw_prp, 
                self.extract_patch_size, 
                self.extract_patch_overlap, 
                multichannel=True
                )
            
            idxs = np.random.choice(
                np.arange(0, len(patches)), size=self.n_patches, replace=False)
            
            for idx in idxs:
                patch_name = (
                    f"{str(path.parent.name)}_"
                    f"preview_patch_{idx:05d}.tif"
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
            
        # Normalize data
        imgs = imgs.astype("float32") / 255
            
        # Setup model
        self.model = Model(parameters=self.model_parameters, model_path=None)
        
        # Train model
        self.model.train(imgs, y=msks)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main  = Main(parameters=main_parameters)
    train = TrainPreview(
        main, parameters=parameters, model_parameters=model_parameters)