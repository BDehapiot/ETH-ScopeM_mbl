#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path
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
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_class", 
        
    # Crops -------------------------------------------------------------------
    
    "crop_size"        : 16,
    "stk_level"        : 1,
    
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "cls",
    "input_shape"        : (None, None, 7),
    "backbone"           : "resnet18", # (sm)
    "filters"            : [8, 16], # (cls & aec)
    "n_classes"          : 6, # (cls)
    "classes"            : ["oth", "epi", "neu", "mac", "bcl", "tcl"],
    "latent_size"        : 128, # (aec)
    "loss"               : "fcl",
    "metric"             : "acc",
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : 16,
    "patch_overlap"      : 0,
    "mask_method"        : "edt", # (sm)

    # Train -------------------------------------------------------------------
    
    "display"            : False,
    "epochs"             : 2048,
    "batch_size"         : 512,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.001,
    "patience"           : 128,
    "balance"            : True, # (cls)
    "use_sample_weights" : False,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : 16384,
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

#%% Class(TrainClasses) -------------------------------------------------------

class TrainClasses:
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
            
#%% Class(TrainClasses) : extract() -------------------------------------------

    def extract_crops(self):
        
        crp_nrm, ncl_cls = [], []
        for p, vws_path in enumerate(self.main.vws_paths):
                    
            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            ncl_cls_path = vws_path / "nuclei_classes.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"
            
            if ncl_cls_path.exists():
                
                print(f"Main() : get_crops() : {vws_path.name}")
            
                # Load data
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)   
                with open(ncl_cls_path, "rb") as file:
                    _ncl_cls = pickle.load(file)    
                stk_nrm = load_compressed_stack(stk_nrm_path)
                
                # Select annotated nuclei
                idxs = np.argwhere(_ncl_cls > 0).squeeze()
                ncl_ctd = {
                    "y" : [ctd[idxs] for ctd in ncl_ctd["y"]],
                    "x" : [ctd[idxs] for ctd in ncl_ctd["x"]],
                    }
                ncl_cls.append(_ncl_cls[idxs])
                
                # Get crops
                crp_nrm.append(get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=self.crop_size, 
                    crop_fraction=1, 
                    stk_level=self.stk_level,
                    ))
                
        # Save
        ncl_cls = np.concatenate(ncl_cls, axis=0)
        with open(self.train_path / "ncl_cls.pkl", "wb") as file:
            pickle.dump(ncl_cls, file)
        crp_nrm = np.concatenate(crp_nrm, axis=0)
        print(crp_nrm.shape)
        save_compressed_stack(crp_nrm, self.train_path / "crp_nrm.npz")
                
#%% Class(TrainClasses) : train() ---------------------------------------------

    def train(self):
        
        # Paths
        ncl_cls_path = self.train_path / "ncl_cls.pkl" 
        crp_nrm_path = self.train_path / "crp_nrm.npz"
        
        # Load data
        with open(ncl_cls_path, "rb") as file:
            ncl_cls = pickle.load(file)    
        crp_nrm = load_compressed_stack(crp_nrm_path)
        
        # Prepare data
        ncl_cls = ncl_cls - 1 # class 1 (other) = 0
        crp_nrm = crp_nrm[:, :7, ...] # remove autofluo channel
        crp_nrm = np.transpose(crp_nrm, (0, 2, 3, 1)) # transposition
        crp_nrm = crp_nrm.astype("float32") / 255 # normalize
        
        self.y = ncl_cls
        self.X = crp_nrm
        
        # Setup model
        self.model = Model(parameters=self.model_parameters, model_path=None)
        
        # Train model
        self.model.train(crp_nrm, y=ncl_cls)
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainClasses(
        main, parameters=parameters, model_parameters=model_parameters)
    
#%% 
    # y = train.y
    # X = train.X
    # _, count = np.unique(y, return_counts=True)
    # cmin = np.min(count)
     
    # bX_trn, by_trn = [], []
    # for c in range(train.model.n_classes):
    #     idxs = np.random.choice(np.argwhere(y == c).squeeze(), size=cmin)
    #     bX_trn.append(X[idxs])
    #     by_trn.append(y[idxs])
    # bX_trn = np.concatenate(bX_trn, axis=0)
    # by_trn = np.concatenate(by_trn, axis=0)
    
#%% Crops ---------------------------------------------------------------------
    
    # # Paths
    # ncl_cls_path = parameters["train_path"] / "ncl_cls.pkl" 
    # crp_nrm_path = parameters["train_path"] / "crp_nrm.npz"    
    
    # # Load data
    # with open(ncl_cls_path, "rb") as file:
    #     ncl_cls = pickle.load(file)    
    # crp_nrm = load_compressed_stack(crp_nrm_path)
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(np.stack(crp_nrm))
        
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