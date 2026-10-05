#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import get_obj_crops

# bdtools
from bdtools.model.model import Model

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------

    "run_crops"          : 0,
    "run_train"          : 1,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_asc_class", 
            
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "cls",
    "input_shape"        : (None, None, 1),
    "loss"               : "fcl",
    "metric"             : "acc",
       
    # # sm
    # "backbone"           : "resnet18",
    
    # cls
    "filters"            : [16, 32],
    "n_classes"          : 2,
    "classes"            : ["Speckle", "Normal"],
    "glob_avg_pool"      : False,
    "regularizer"        : None,
    "dropout"            : 0.3,
    
    # # aec
    # "filters"            : [16, 32],
    # "latent_size"        : 128,
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : 12,
    "patch_overlap"      : 0,

    # Train -------------------------------------------------------------------
    
    "display"            : 0,
    "epochs"             : 2048,
    "batch_size"         : 32,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.001,
    "patience"           : 512,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : 8192,
    "augment_gamma_p"    : 0.0,
    "augment_gblur_p"    : 0.0,
    "augment_noise_p"    : 0.0,
    "augment_flip_p"     : 0.5,
    "augment_distort_p"  : 0.0,
    "augment_balance"    : True, # (cls)
    
    "augment_parameters" : {
        
    # Gamma
    "gamma_low"          : 0.75,
    "gamma_high"         : 1.25,
    "gamma_chn"          : "independent",
    
    # Gaussian blur
    "gblur_sigma_low"    : 1,
    "gblur_sigma_high"   : 2,
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

#%% Class(TrainASCClasses) ----------------------------------------------------

class TrainASCClasses:
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
                        
#%% Class(TrainASCClasses) : extract_crops() ----------------------------------

    def extract_crops(self):

        asc_crp = []
        asc_ann = {"y" : [], "x" : [], "class" : []}
        for p, vws_path in enumerate(self.main.vws_paths):
                                
            # Paths
            asc_ann_path = vws_path / "asc_annotations.pkl"
            stk_nrm_path = vws_path / "stack_norm.tif"
            
            if asc_ann_path.exists():
                
                # Log
                t0i = time.time()
                log_str = f"TrainASCClasses() : extract_crops() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
            
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(asc_ann_path, "rb") as file:
                    _asc_ann = pickle.load(file)    
                stk_nrm = io.imread(stk_nrm_path)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                # Append asc_ann
                asc_ann["y"].extend(_asc_ann["y"])
                asc_ann["x"].extend(_asc_ann["x"])
                asc_ann["class"].extend(_asc_ann["class"])
                                
                # Get crops
                asc_crp.append(get_obj_crops(
                    stk_nrm[4, ...], _asc_ann["y"], _asc_ann["x"], 
                    crop_size=self.model_parameters["patch_size"], 
                    crop_fraction=None, 
                    ))
                                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
        # Save ----------------------------------------------------------------
                
        # Log
        log_str = "TrainASCClasses() : extract_crops() : save all"
        print("-" * len(log_str))
        print(log_str)
        print("-" * len(log_str))
                
        # Save
        t0 = time.time()

        with open(self.train_path / "asc_ann.pkl", "wb") as file:
            pickle.dump(asc_ann, file)
        asc_crp = np.concatenate(asc_crp, axis=0)
        io.imsave(self.train_path / "asc_crp.tif", 
            asc_crp, check_contrast=False,
            )
        
        t1 = time.time()
        print(f"> {t1 - t0:.3f}s")
        
#%% Class(TrainASCClasses) : train() ------------------------------------------

    def train(self):
        
        # Paths
        asc_ann_path = self.train_path / "stock_in" / "asc_ann.pkl"
        asc_crp_path = self.train_path / "stock_in" / "asc_crp.tif"
        
        # Load
        with open(asc_ann_path, "rb") as file:
            asc_ann = pickle.load(file)    
        asc_crp = io.imread(asc_crp_path)
        
        # Prepare data
        asc_ann = np.array(asc_ann["class"]) - 1
        asc_crp = asc_crp.astype("float32") / 255        
        
        # Setup model
        self.model = Model(parameters=self.model_parameters, model_path=None)
        
        # Train model
        self.model.train(asc_crp, y=asc_ann)
                        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainASCClasses(
        main, parameters=parameters, model_parameters=model_parameters)  