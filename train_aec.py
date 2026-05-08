#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from main import Main, main_parameters

# functions
from functions import (
    get_crops, save_compressed_stack, load_compressed_stack)

# bdtools
from bdtools.model.model import Model

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_crops"        : 0,
    "run_train"        : 1,
        
    # Crops -------------------------------------------------------------------
    
    "crop_size"        : 32,
    "crop_fraction"    : 0.01,
    "stk_level"        : 1,
    
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "aec",
    "input_shape"        : (None, None, 7),
    "filters"            : [32, 64, 128], # (aec)
    "latent_size"        : 128, # (aec)
    "activation"         : "sigmoid",
    "loss"               : "mse",
    "metric"             : "mae",
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : 32,
    "patch_overlap"      : 0,

    # Train -------------------------------------------------------------------
    
    "display"            : 0,
    "epochs"             : 512,
    "batch_size"         : 32,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.0001,
    "patience"           : 128,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : None,
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

#%% Class(TrainAEC) -----------------------------------------------------------

class TrainAEC:
    def __init__(self, main, parameters=None, model_parameters=None):
        self.main = main
        self.parameters = parameters
        self.model_parameters = model_parameters
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
        self.model = Model(parameters=self.model_parameters, model_path=None)
        
        # Train model
        self.model.train(crp_nrm, y=None)
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(
        main, parameters=parameters, model_parameters=model_parameters)
    
#%% Predict
    
    # # Parameters
    # subset = 0.01
    # model_name = "model_32_2_512_59273"

    # # Paths
    # crp_nrm_path = main.data_path / "crops_raw.npz"
    
    # # Load model
    # model = AutoEncoder(
    #     parameters=None, 
    #     model_path=Path.cwd() / model_name
    #     )
    
    # # Load data
    # crp_nrm = load_compressed_stack(crp_nrm_path)
    # crp_num = crp_nrm.shape[0]   
    
    # # Select data
    # idxs = np.random.choice(
    #     np.arange(0, crp_num), size=int(crp_num * subset), replace=False)
    # X = crp_nrm[idxs][:, :7, ...]
    # X = np.transpose(X, (0, 2, 3, 1)) # transposition
    
    # # Normalize data
    # X = X.astype("float32") / 255
    
    # # Predict
    # prd = model.predict(X)
    # prd = np.transpose(prd, (0, 3, 1, 2))
    # X = np.transpose(X, (0, 3, 1, 2))
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(X)
    # vwr.add_image(prd)