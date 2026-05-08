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

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_patches_0" : 0, # new
    "run_patches_1" : 0, # from preexisting masks
    "run_annotate"  : 0,
    "run_train"     : 1,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_nuclei", 
    
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

#%% Class(TrainNuclei) --------------------------------------------------------

class TrainNuclei:
    def __init__(self, main, parameters=None, model_parameters=None):
        self.main = main
        self.parameters = parameters
        self.model_parameters = model_parameters
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

#%% Class(TrainNuclei) extract_patches() --------------------------------------

    def extract_patches_0(self):
        """ to be implemented """
        pass
                
    def extract_patches_1(self):
        
        """Broken"""

        msk_paths = list(self.train_path.glob("*_mask.tif"))
        
        for p, msk_path in enumerate(msk_paths):
            
            # Parse paths
            qpt_path = self.main.qpt_paths[p]
            mtd_path = self.main.mtd_paths[p]
            vws_path = self.main.vws_paths[p]
            pvw_its_path = vws_path / "preview_intensities.pkl"
            roi_idx  = int(msk_path.stem.split("_")[-5])
            patch_idx = int(msk_path.stem.split("_")[-2])
            
            print(roi_idx)
            print(patch_idx)
                        
            # Load
            with open(mtd_path, "rb") as file:
                mtd = pickle.load(file)
            with open(pvw_its_path, "rb") as file:
                pvw_its = pickle.load(file)
            ncl = load_view(
                qpt_path, metadata=mtd, channels="DAPI", 
                level=0, roi_idx=roi_idx,
                )
            ncl = norm_data(ncl, pvw_its, channel_idx=0)
            
            # Extract patch
            
            patches = extract_patches(
                ncl, self.extract_patch_size, self.extract_patch_overlap)
            
            print(len(patches))
            
            patch = patches[patch_idx]
            patch = (patch * 255).astype("uint8")
            
            # Save
            patch_name = str(msk_path).replace("_mask", "")
            io.imsave(
                self.train_path / patch_name,
                patch, check_contrast=False,
                )
                            
#%% Class(TrainNuclei) annotate() ---------------------------------------------

    def annotate(self):
        Annotate(self.train_path)  
        
#%% Class(TrainNuclei) train() ------------------------------------------------
    
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
    train = TrainNuclei(
        main, parameters=parameters, model_parameters=model_parameters)