#%% Imports -------------------------------------------------------------------

import gc
import pickle
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    load_metadata, load_arr,
    
    )

# bdtools
from bdtools.norm import norm_pct
from bdtools.conn import lbl_conn
from bdtools.unet.unet import UNet

# skimage
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops_table
from skimage.segmentation import watershed, expand_labels

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path"     : Path("D:\local_mbl\data"),
    "model_name"    : "model_256_edt_512-28",

    # extract()
    "preview_level" : 4,
    
    # process()
    "process_level" : 1,
    "thresh_0"      : 0.25,
    "thresh_1"      : 0.50,
        
    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.initialize()
        self.extract()
        self.process()
        # self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        self.qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.sld_paths = [
            self.data_path / f"{path.stem[:12]}"
            for path in self.qpt_paths
            ]
       
        # Setup slide directory
        for sld_path in self.sld_paths:
            if not sld_path.exists():
                sld_path.mkdir(parents=True, exist_ok=True) 

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        self.view_paths = [] 
        for p, path in enumerate(self.qpt_paths): 
            sld_path = self.sld_paths[p]
            mtd_path = sld_path / "metadata.pkl"
            pvw_path = sld_path / f"preview_level-{self.preview_level}.tif"
            
            # Metadata
            if not mtd_path.exists():
                print(f"extract() : {path.name}")
                mtd = load_metadata(path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
        
            # Preview
            if not pvw_path.exists():
                pvw = load_arr(
                    path, metadata=mtd, 
                    channels="all", level=self.preview_level
                    )
                pvw = np.mean(pvw, axis=0)
                pvw = (norm_pct(pvw) * 255).astype("uint8")
                io.imsave(pvw_path, pvw, check_contrast=False)

            # View(s)
            for idx in range(len(mtd["roi"])):
                
                # Setup view directory
                view_path = sld_path / f"view_{idx:02d}"
                self.view_paths.append(view_path)
                if not view_path.exists():
                    view_path.mkdir(parents=True, exist_ok=True) 
                
                # DAPI at level 0 (for predictions)
                dapi_path = view_path / "dapi_level-0.tif"
                if not dapi_path.exists():
                    dapi = load_arr(
                        path, metadata=mtd, 
                        channels="DAPI", level=0, roi_idx=idx,
                        )
                    dapi = (norm_pct(dapi) * 255).astype("uint8")
                    io.imsave(dapi_path, dapi, check_contrast=False)
                    
                # Stack at process level 
                stk_path = view_path / f"stack_level-{self.process_level}.tif"
                if not stk_path.exists():
                    stk = load_arr(
                        path, metadata=mtd, 
                        channels="all", level=self.process_level, roi_idx=idx,
                        )
                    stk = ((stk + 8) / 16 ).astype("uint8") # 12bits to 8bits
                    io.imsave(stk_path, stk, check_contrast=False)                               
                        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):

        for p, path in enumerate(self.view_paths):
            dap_path = path / "dapi_level-0.tif"
            msk_path = path / f"mask_level-{self.process_level}.tif"

            if not msk_path.exists():
                print(f"process() : {path.parent.name}/{path.name}")
                
                if "unet" not in locals():
                    unet = UNet(
                        parameters=None, model_path=Path.cwd() / self.model_name)
                
                dap = io.imread(dap_path)
                dap = dap.astype("float32") / 255
            
                # Predict
                prd = unet.predict(dap)
                dscale_factor = 2 ** self.process_level
                prd = downscale_local_mean(prd, dscale_factor)
                
                # Get mask
                msk_0 = prd > self.thresh_0
                msk_1 = prd > self.thresh_1
                msk = watershed(-prd, label(msk_1), mask=msk_0)
                msk[lbl_conn(msk, conn=2) == 2] = 0
                msk = remove_small_objects(
                    msk, min_size=0.5 * (2 ** self.process_level) ** 2)
                msk = ((msk > 0) * 255).astype("uint8")
                              
                # Save & clear
                io.imsave(msk_path, msk, check_contrast=False)
                tf.keras.backend.clear_session()
                gc.collect()
            
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        for p, path in enumerate(self.view_paths):
            msk_path = path / f"mask_level-{self.process_level}.tif"
            stk_path = path / f"stack_level-{self.process_level}.tif"
            results_path = path / f"results_level-{self.process_level}.csv"
            
            if not results_path.exists():
                print(f"analyse() : {path.parent.name}/{path.name}")
                
                msk = io.imread(msk_path) > 0
                stk = io.imread(stk_path)
                
                # Measure intensities
                msk = expand_labels(
                    label(msk), distance=1 * (2 ** self.process_level) ** 2)
                results = regionprops_table(
                    msk, intensity_image=np.moveaxis(stk, 0, -1), 
                    properties=("label", "intensity_mean"),
                    )
                results.pop("label", None)
                results = {
                    channel : value 
                    for channel, value 
                    in zip(self.channels, results.values())
                    }
                results = pd.DataFrame(results)
                results.to_csv(results_path, index=False)
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)

#%%

    import time
    
    # -------------------------------------------------------------------------

    view_idx = 0
    channels = main_parameters["channels"]
    distance = 1 * (2 ** main.process_level) ** 2
    msk_path = main.view_paths[view_idx] / f"mask_level-{main.process_level}.tif"
    stk_path = main.view_paths[view_idx] / f"stack_level-{main.process_level}.tif"
    
    msk = io.imread(msk_path)
    stk = io.imread(stk_path)
    
    # -------------------------------------------------------------------------

    t0 = time.time()
    print("expand_labels() : ", end="", flush=False)
    
    msk = expand_labels(label(msk), distance=distance)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("regionprops_table() : ", end="", flush=False)

    results = regionprops_table(
        msk, intensity_image=np.moveaxis(stk, 0, -1), 
        properties=("label", "intensity_mean"),
        )
    results.pop("label", None)
    results = {
        channel : value 
        for channel, value in zip(channels, results.values())
        }
    results = pd.DataFrame(results)
    results.to_csv(Path.cwd() / "results.csv", index=False)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_labels(msk)
    vwr.add_image(stk)