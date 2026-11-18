#%% Imports -------------------------------------------------------------------

from main import Main

#%% Parameters ----------------------------------------------------------------

main_parameters = {

    # Procedure ---------------------------------------------------------------
    
    "run_extract"         : 1, 
    "run_predict_nuclei"  : 1,
    "run_predict_asc"     : 1,
    "run_classify_nuclei" : 1,
    "run_classify_asc"    : 1,
    "run_measure"         : 1,
    "run_analyse"         : 1,
    
    # 0 to skip, 1 to run if not already done and 2 to force execution.
    
    # Paths -------------------------------------------------------------------
        
    "data_path"     : "D:\local_mbl\data", # Path to data directory

    # Segmentation ------------------------------------------------------------
    
    # Nuclei
    "ncl_thresh_0"  : 0.20, # probability thresh. 0 (outlines)
    "ncl_thresh_1"  : 0.40, # probability thresh. 1 (cores)
    "ncl_min_size"  : 8,    # min. area (in µm² at level = 0)
    "ncl_max_size"  : 512,  # max. area (in µm² at level = 0)
     
    # ASC
    "asc_thresh_0"  : 0.1,  # probability thresh. 0 (outlines)
    "asc_thresh_1"  : None, # probability thresh. 1 (cores)
    "asc_min_size"  : 8,    # min. area (in µm² at level = 0)
    "asc_max_size"  : None, # max. area (in µm² at level = 0)
    
    # 2 thresholds approach (object separation) works for edt models only 
    # Use None for thresh_1 with binary models.
    
    # Measurments -------------------------------------------------------------
    
    # Nuclei fluo. intensities
    "pixel_size"    : 0.5,  # image resolution (in µm at level = 0)
    "int_crop_size" : 16,   # fluo. intensities window size (in µm at level = 0)
    "int_norm"      : True, # normalize fluo. intensities (per channel)
    
    # Fluorescence channel intensities are measured in a square region of size 
    # "int_crop_size" around the nucleus centroid. If normalized, measured 
    # values dividing them by the average value across all nuclei in the same
    # sample.
    
    # Nuclei distance to ASC
    "dst_bin_size"  : 4,    # distance bin size (in µm at level = 0)
    "dst_bin_max"   : 32,   # distance bin max (in µm at level = 0)
    "dst_norm"      : True, # normalize distances (per class)
    
    # Nuclei distances to the closest ASC cluster (non-speckle or speckle) are
    # measured and binned according to "dst_bin_size" and "dst_norm". If 
    # normalized, the measured values are divided by the overall baseline class
    # frequencies across all bins to compute spatial enrichment.

    # Analyse() ---------------------------------------------------------------
        
    "skip_views" : [
        "251127_D2_S2_Scan1_unmixed_01",
        ], 
    
    # Add name of the view(s) that will be skipped during the merged analysis.
    
    # Colors ------------------------------------------------------------------
        
    "ncl_colors" : [
        [0.0, 0.0, 1.0], # Other
        [0.0, 0.8, 0.2], # Epithelium
        [1.0, 0.0, 0.0], # Neutrophil
        [1.0, 0.0, 1.0], # Macrophage
        [0.0, 1.0, 1.0], # Bcell
        [0.8, 0.8, 0.0], # Tcell
        ],

    # Plot  -------------------------------------------------------------------
        
    # Error bars
    "err_type" : "sem", # use "sem" or "std"
    
    # Line & grids
    "hline_params" : {
        "y" : 1, "color" : "0.75", "zorder" : 0,
        "linewidth" : 0.5, "linestyle" : ":",
        },
    
    "grid_params" : {
        "axis" : "x", "color" : "0.75", "zorder" : 0,
        "linewidth" : 0.5, "linestyle" : ":", 
        },

#%% Advanced parameters -------------------------------------------------------

    # ... Should not be modified ...
        
    # Select model ------------------------------------------------------------

    "model_pvw_name"       : "model-sm_preview_256_binary_227-1024",
    "model_ncl_name"       : "model-sm_nuclei_256_edt_192-1024",
    "model_asc_name"       : "model-sm_asc_256_binary_377-1024",
    "model_ncl_class_name" : "model-cls_nuclei_12_2_6_5045-8190b",
    "model_asc_class_name" : "model-cls_asc_48_4_3_4467-8190b",

    # Pyramid levels ----------------------------------------------------------
    
    "pvw_level" : 4,
    "stk_level" : 1,
    "ncl_level" : 0,
    "asc_level" : 0,
        
    # Maps  -------------------------------------------------------------------
    
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"],
    "ncl_classes" : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell"],
    "asc_classes" : [
        "Speckle", "Non-speckle", "Non-valid"],
    "plot_channels" : [
        "Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"],
    "plot_classes"  : [
        "Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"],
    
    }

#%% Outputs -------------------------------------------------------------------


#%%

if __name__ == "__main__":
    main = Main(parameters=main_parameters, run=True)