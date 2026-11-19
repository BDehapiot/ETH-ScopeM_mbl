#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import imread_channel, get_obj_crops

# bdtools
from bdtools.model.model import Model

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_crops"          : 0,
    "run_train"          : 1,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_nuclei_class", 
        
    # Crops -------------------------------------------------------------------
    
    "crop_size"          : 12,
    
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "cls",
    "input_shape"        : (None, None, 5),
    "loss"               : "cce",
    "metric"             : "acc",
       
    # # sm
    # "backbone"           : "resnet18",
    
    # cls
    "filters"            : [16, 32],
    "n_classes"          : 6,
    "classes"            : ["oth", "epi", "neu", "mac", "bcl", "tcl"],
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

#%% Class(TrainNucleiClasses) -------------------------------------------------

class TrainNucleiClasses:
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
            
#%% Class(TrainNucleiClasses) : extract_crops() -------------------------------

    def extract_crops(self):

        ncl_crp = []
        ncl_ann = {"y" : [], "x" : [], "class" : []}
        for p, vws_path in enumerate(self.main.vws_paths):
                                
            # Paths
            ncl_ann_path = vws_path / "nuclei_annotations.pkl"
            stk_nrm_path = vws_path / "stack_norm.tif"
            
            if ncl_ann_path.exists():
                
                # Log
                t0i = time.time()
                log_str = f"TrainNucleiClasses() : extract_crops() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
            
                # Load --------------------------------------------------------
                
                t0 = time.time()
                
                with open(ncl_ann_path, "rb") as file:
                    _ncl_ann = pickle.load(file)  
                stk_nrm = imread_channel(
                    stk_nrm_path, channel_idx=[1, 2, 3, 5, 6]
                    )  
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - load")
                                
                # Get crops ---------------------------------------------------
                
                t0 = time.time()
                
                # Append ncl_ann
                ncl_ann["y"].extend(_ncl_ann["y"])
                ncl_ann["x"].extend(_ncl_ann["x"])
                ncl_ann["class"].extend(_ncl_ann["class"])
                                
                # Get crops
                ncl_crp.append(get_obj_crops(
                    stk_nrm, _ncl_ann["y"], _ncl_ann["x"], 
                    crop_size=self.model_parameters["patch_size"], 
                    crop_fraction=None, 
                    ))
                                                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s - get crops")
                
                t1f = time.time()
                print(f"> {t1f - t0i:.3f}s")
                
        # Save ----------------------------------------------------------------
                
        # Log
        log_str = "TrainNucleiClasses() : extract_crops() : save all"
        print("-" * len(log_str))
        print(log_str)
        print("-" * len(log_str))
                
        # Save
        t0 = time.time()

        with open(self.train_path / "ncl_ann.pkl", "wb") as file:
            pickle.dump(ncl_ann, file)
        ncl_crp = np.concatenate(ncl_crp, axis=0)
        io.imsave(self.train_path / "ncl_crp.tif", 
            ncl_crp, check_contrast=False,
            )
        
        t1 = time.time()
        print(f"> {t1 - t0:.3f}s")
        
#%% Class(TrainNucleiClasses) : train() ---------------------------------------
        
    def train(self):
        
        # Paths
        ncl_ann_paths = sorted(self.train_path.glob("ncl_ann*.pkl"))
        ncl_crp_paths = sorted(self.train_path.glob("ncl_crp*.tif"))
        if not ncl_ann_paths or not ncl_crp_paths:
            raise FileNotFoundError(
                "No ncl_ann*.pkl or ncl_crp*.tif files found")
    
        # Load
        ncl_ann = []
        for path in ncl_ann_paths:
            with open(path, "rb") as file:
                _ncl_ann = pickle.load(file)
                ncl_ann.append(np.array(_ncl_ann["class"]) - 1)
        ncl_crp = [
            io.imread(path).astype("float32") / 255 for path in ncl_crp_paths]
        
        # Concatenate
        ncl_ann = np.concatenate(ncl_ann, axis=0)
        ncl_crp = np.concatenate(ncl_crp, axis=0)
    
        # Setup model
        self.model = Model(parameters=self.model_parameters, model_path=None)
    
        # Train model
        self.model.train(ncl_crp, y=ncl_ann)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from main import Main
    from run import main_parameters  
    main = Main(parameters=main_parameters, run=False)
    train = TrainNucleiClasses(
        main, parameters=parameters, model_parameters=model_parameters)  
