#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import (
    get_crops, prepare_crops, 
    save_compressed_stack, load_compressed_stack
    )

# bdtools
from bdtools.model.model import Model

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_crops"          : 0,
    "run_train"          : 0,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_class", 
        
    # Crops -------------------------------------------------------------------
    
    "crop_size"          : 12,
    "stk_level"          : 1,
    
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    # Global
    "model_type"         : "cls",
    "input_shape"        : (None, None, 5),
    "loss"               : "cce",
    "metric"             : "acc",
    
    # cls
    "filters"            : [16, 32],
    "n_classes"          : 6,
    "classes"            : ["oth", "epi", "neu", "mac", "bcl", "tcl"],
    "glob_avg_pool"      : False,
    "regularizer"        : None,
    "dropout"            : 0.3,
    
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
            ncl_cls_path = vws_path / "nuclei_classes_annotations.pkl"
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
                    crop_fraction=None, 
                    stk_level=self.stk_level,
                    ))
                
        # Save
        ncl_cls = np.concatenate(ncl_cls, axis=0)
        with open(self.train_path / "ncl_cls.pkl", "wb") as file:
            pickle.dump(ncl_cls, file)
        crp_nrm = np.concatenate(crp_nrm, axis=0)
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
        crp_nrm = prepare_crops(crp_nrm)
        
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
    
#%% Predict -------------------------------------------------------------------

    # Paths
    vws_idx = 15
    vws_path = main.vws_paths[vws_idx]
    stk_nrm_path = vws_path / "stack_norm.npz" 
    ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    
    # Load model
    model_path = Path(Path.cwd(), "model-cls_12_2_6_5045-8190b")
    model = Model(parameters=None, model_path=model_path)
    
    # Load data
    with open(ncl_ctd_path, "rb") as file:
        ncl_ctd = pickle.load(file)   
    stk_nrm = load_compressed_stack(stk_nrm_path)
    
    # Get crops
    crp_nrm = get_crops(
        stk_nrm, ncl_ctd, 
        crop_size=parameters["crop_size"],
        stk_level=parameters["stk_level"],
        )
    crp_nrm = prepare_crops(crp_nrm)
        
    # Predict
    prds = model.predict(crp_nrm)  
    prds = np.argmax(prds, axis=1)
    prds += 1

#%% Display -------------------------------------------------------------------
    
    import napari
    
    # Maps
    
    colors = [
        "gray", "yellow", "green",
        "red", "orange",
        "cyan", "magenta",
        ]
    
    names = [
        "NaN", "Other", "Epithelium",
        "Neutrophil", "Macrophage", 
        "B cell", "T cell", 
        ]
    
    class_map = {
        "cls_nan" : {"tag" : 0, "color" : colors[0], "name" : names[0]},
        "cls_oth" : {"tag" : 1, "color" : colors[1], "name" : names[1]},
        "cls_epi" : {"tag" : 2, "color" : colors[2], "name" : names[2]},
        "cls_neu" : {"tag" : 3, "color" : colors[3], "name" : names[3]},
        "cls_mac" : {"tag" : 4, "color" : colors[4], "name" : names[4]},
        "cls_bcl" : {"tag" : 5, "color" : colors[5], "name" : names[5]},
        "cls_tcl" : {"tag" : 6, "color" : colors[6], "name" : names[6]},
        }
    
    color_map = {
        v["tag"]: v["color"] for v in class_map.values()}
    
    ncl_ctd = np.column_stack((
        ncl_ctd["y"][parameters["stk_level"]],
        ncl_ctd["x"][parameters["stk_level"]],
        ))
    
    # -------------------------------------------------------------------------    
    
    vwr = napari.Viewer()
    vwr.add_image(
        stk_nrm, name="stk_nrm", visible=1,
        gamma=0.75, 
        )
            
    vwr.add_points(
        ncl_ctd, name="points", visible=1,
        blending="translucent", size=8, border_width=0.0,
        features={"class_label": prds},
        face_color="class_label",
        face_color_cycle=color_map,
        opacity=0.5, 
        )
