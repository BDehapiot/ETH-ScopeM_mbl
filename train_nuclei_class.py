#%% Imports -------------------------------------------------------------------

from pathlib import Path

#%% Comments ------------------------------------------------------------------

'''
This code is not yet updated. Please refer to "train_asc_class.py" to implement
it. The parameters listed below were used to train the currently used nuclei 
classification model : model-cls_nuclei_12_2_6_5045-8190b
Note : the "annotate_ncl.py" code was however already updated
'''

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_crops"          : 0,
    "run_classify"       : 0,
    "run_train"          : 0,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_ncl_class", 
        
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
