#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%%
    
    # data_path = Path("D:\local_mbl\data")
    # dap_paths = list(data_path.rglob("dapi_level-0.tif"))
    
    # daps = []
    # for path in dap_paths:
    #     daps.append(io.imread(path))
        
    # stats = {
    #     "name" : [],
    #     "min"  : [],
    #     "max"  : [],
    #     "avg"  : [],
    #     "med"  : [],
    #     "qt0"  : [],
    #     "qt1"  : [],
    #     }
    
    # for dap in daps:
    #     msk = dap > 0
    #     stats["name"].append(f"{path.parent.parent.name}/{path.parent.name}")
    #     stats["min" ].append(np.min(dap[msk]))
    #     stats["max" ].append(np.max(dap[msk]))
    #     stats["avg" ].append(np.mean(dap[msk]))
    #     stats["med" ].append(np.median(dap[msk]))
    #     stats["qt0" ].append(np.quantile(dap[msk], 0.1))
    #     stats["qt1" ].append(np.quantile(dap[msk], 0.9))
    
    # stats = pd.DataFrame(stats)
    
    # plt.scatter(stats["avg" ], stats["med" ])
    
#%%

    # channel = 4 
    # data_path = Path("D:\local_mbl\data")
    # stk_paths = list(data_path.rglob("stack_level-1.tif"))
    
    # imgs = []
    # for path in stk_paths:
    #     imgs.append(io.imread(path)[channel])
        
    # min_y = np.min([img.shape[0] for img in imgs])
    # min_x = np.min([img.shape[1] for img in imgs])
    
    # imgs_crop = []
    # for img in imgs:
    #     y_pad = img.shape[0] - min_y
    #     x_pad = img.shape[1] - min_x
    #     y0 = y_pad // 2
    #     y1 = img.shape[0] - y_pad // 2
    #     x0 = x_pad // 2
    #     x1 = img.shape[1] - x_pad // 2
    #     imgs_crop.append(img[y0:y1, x0:x1])
        
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(np.stack(imgs_crop))
    
#%% 

    data_path = Path("D:\local_mbl\data")
    mtd_paths = list(data_path.rglob("metadata.pkl"))
    
    mtds = []
    for path in mtd_paths:
        with open(path, "rb") as file:
            mtds.append(pickle.load(file))
    
    channel = 0
    measure = "q75"
    vals = []
    for mtd in mtds:
        for idx in range(len(mtd["roi"])):
            vals.append(mtd["stats"][idx][measure][channel])