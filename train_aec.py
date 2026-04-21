#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import (
    get_crops, save_compressed_stack, load_compressed_stack)

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model

#%% Inputs --------------------------------------------------------------------

aec_parameters = {
    
    # Procedure
    "run_crops"     : 1,
    "run_train"     : 0,
    
    # get_crops() 
    "crop_size"     : 32,
    "crop_fraction" : 0.05,
    "stk_level"     : 1,
    
    # # build()
    # "n_channels"    : 8,
    # "latent_size"   : 64,
    
    # # train()
    # "batch_size"    : 256,
    
    }

#%% Class(TrainAEC) -----------------------------------------------------------

class TrainAEC:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_crops:
            self.extract_crops()
        if self.run_train:
            self.build()
            self.train()
            
#%% Class(TrainAEC) : extract() -----------------------------------------------

    def extract_crops(self):
        
        crp_raw = []
        for p, vws_path in enumerate(self.main.vws_paths):
            
            print(f"Main() : get_crops() : {vws_path.name}")
            
            # Paths
            stk_raw_path = vws_path / "stack_raw.npz"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            crp_raw_path = vws_path / "crops_raw.pkl"
            
            # Load data
            stk_raw = load_compressed_stack(stk_raw_path)
            with open(ncl_ctd_path, "rb") as file:
                ncl_ctd = pickle.load(file)
                
            # Get crops
            crp_raw.append(get_crops(
                stk_raw, ncl_ctd, 
                crop_size=self.crop_size, 
                crop_fraction=self.crop_fraction, 
                stk_level=self.stk_level,
                ))
            
        # Save
        crp_raw = np.concatenate(crops, axis=0)
        save_compressed_stack(crops, crp_raw_path)
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(main, parameters=aec_parameters)
    crops = train.crops
    
    crops = np.concatenate(crops, axis=0)