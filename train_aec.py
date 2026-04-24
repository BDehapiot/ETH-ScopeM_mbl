#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import (
    get_crops, save_compressed_stack, load_compressed_stack)

# bdtools
from bdtools.autoencoder.autoencoder import AutoEncoder

#%% Inputs --------------------------------------------------------------------

autoencoder_parameters = {
    
    # Procedure
    "run_crops"        : 0,
    "run_train"        : 1,
        
    # get_crops() 
    "crop_size"        : 32,
    "crop_fraction"    : 0.01,
    "stk_level"        : 1,
    
    # model -------------------------------------------------------------------

    # Paths
    "root_path"        : None,
    "model_name"       : None,
        
    # Build
    "input_shape"      : (32, 32, 7),
    "filters"          : [32, 64],
    "latent_size"      : 512,
    "activation"       : "sigmoid",
    
    # Train
    "epochs"           : 256,
    "batch_size"       : 256,
    "validation_split" : 0.2,
    "metric"           : "soft_dice_coef",
    "learning_rate"    : 0.001,
    "patience"         : 64,
        
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
            self.train()
            
#%% Class(TrainAEC) : extract() -----------------------------------------------

    def extract_crops(self):
        
        crp_nrm = []
        for p, vws_path in enumerate(self.main.vws_paths):
            
            print(f"Main() : get_crops() : {vws_path.name}")
            
            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"
            crp_nrm_path = self.main.data_path / "crops_raw.npz"
            
            # Load data
            with open(ncl_ctd_path, "rb") as file:
                ncl_ctd = pickle.load(file)
            stk_nrm = load_compressed_stack(stk_nrm_path)

            # Get crops
            crp_nrm.append(get_crops(
                stk_nrm, ncl_ctd, 
                crop_size=self.crop_size, 
                crop_fraction=self.crop_fraction, 
                stk_level=self.stk_level,
                ))
            
        # Save
        crp_nrm = np.concatenate(crp_nrm, axis=0)
        save_compressed_stack(crp_nrm, crp_nrm_path)
                
#%% Class(TrainAEC) : train() -------------------------------------------------

    def train(self):
        
        # Paths
        crp_nrm_path = self.main.data_path / "crops_raw.npz"
        
        # Load data
        crp_nrm = load_compressed_stack(crp_nrm_path)
        crp_nrm = crp_nrm[:, :7, ...] # remove autofluo channel
        crp_nrm = np.transpose(crp_nrm, (0, 2, 3, 1)) # transposition
        
        # Normalize data
        crp_nrm = crp_nrm.astype("float32") / 255
        
        # Setup model
        self.autoencoder = AutoEncoder(
            parameters=self.parameters, model_path=None)
        
        # Train model
        self.autoencoder.train(crp_nrm)
        
        pass
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(main, parameters=autoencoder_parameters)
    
#%% Predict
    
    # # Parameters
    # subset = 0.01

    # # Paths
    # crp_nrm_path = main.data_path / "crops_raw.npz"
    
    # # Load
    # crp_nrm = load_compressed_stack(crp_nrm_path)
    # crp_num = crp_nrm.shape[0]   
    
    # # Select data
    # idxs = np.random.choice(
    #     np.arange(0, crp_num), size=int(crp_num * subset), replace=False)
    # X = crp_nrm[idxs][:, :7, ...]
    
    # # Predict
    # prd = train.predict(X)
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(X)
    # vwr.add_image(prd)