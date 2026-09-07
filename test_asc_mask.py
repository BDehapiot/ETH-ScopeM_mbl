#%% Imports -------------------------------------------------------------------

from pathlib import Path
from main import Main, main_parameters

#%% Inputs --------------------------------------------------------------------

test_parameters = {
    
    # Procedure
    "force_predict_asc" : 2,
     
    # predict_asc()
    "asc_level"         : 0,
    "asc_thresh"        : 0.2,
    "min_dist"          : 5,   
    "min_int"           : 0.08, 
    "radius"            : 7, 
    "sigma"             : 1.0, 
    "min_corr"          : 0.9, 
    
    }

main_parameters.update(test_parameters)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)    

    # # Parameters
    # vws_idx = 0
    
    # # Fetch
    # vws_path = main.vws_paths[vws_idx]
    # pvw_msk_path = vws_path / "preview_mask.tif"
    # pvw_msk = io.imread(pvw_msk_path)