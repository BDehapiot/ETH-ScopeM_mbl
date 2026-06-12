#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import load_view, norm_data

# bdtools
from bdtools.model.model import Model
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

# skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_patches"  : 0,
    "run_annotate" : 0,
    "run_train"    : 0,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_asc", 
    
    # Extract -----------------------------------------------------------------
    
    ""
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
    "mask_method"        : "edt",

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
    "augment_balance"    : False, # (cls)
    
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

#%% Class(TrainASC) -----------------------------------------------------------

class TrainASC:
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
            
#%% Class(TrainASC) extract_patches() -----------------------------------------

    def extract_patches(self):
        
        for p, vws_path in enumerate(self.main.vws_paths):
            
            if p == 0:
            
                # Paths
                qpt_path = self.main.qpt_paths[p]
                mtd_path = self.main.mtd_paths[p]
                roi_idx = int(vws_path.stem[-2:])
                
                # Load
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                stk_raw = load_view(
                    qpt_path, metadata=mtd, channels="ASC", 
                    level=0, roi_idx=roi_idx,
                    )
                
                # Extract patches
                patches = extract_patches(
                    stk_raw, 
                    self.extract_patch_size, 
                    self.extract_patch_overlap, 
                    multichannel=False
                    )
                
                # Get random indexes
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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main  = Main(parameters=main_parameters)
    train = TrainASC(
        main, parameters=parameters, model_parameters=model_parameters)
