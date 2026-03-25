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
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"],
    
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
        self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        self.qptiff_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.slide_paths = [
            self.data_path / f"{path.stem[:12]}"
            for path in self.qptiff_paths
            ]
       
        # Setup slide directory
        for slide_path in self.slide_paths:
            if not slide_path.exists():
                slide_path.mkdir(parents=True, exist_ok=True) 

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        self.view_paths = [] 
        for p, path in enumerate(self.qptiff_paths): 
            slide_path    = self.slide_paths[p]
            metadata_path = slide_path / "metadata.pkl"
            preview_path  = slide_path / f"preview_level-{self.preview_level}.tif"
            
            # Metadata
            if not metadata_path.exists():
                print(f"extract() : {path.name}")
                metadata = load_metadata(path)
                with open(metadata_path, "wb") as file:
                    pickle.dump(metadata, file)
            else:
                with open(metadata_path, "rb") as file:
                    metadata = pickle.load(file)
        
            # Preview
            if not preview_path.exists():
                preview = load_arr(
                    path, metadata=metadata, 
                    channels="all", level=self.preview_level
                    )
                preview = np.mean(preview, axis=0)
                preview = (norm_pct(preview) * 255).astype("uint8")
                io.imsave(preview_path, preview, check_contrast=False)

            # View(s)
            for idx in range(len(metadata["roi"])):
                
                # Setup view directory
                view_path = slide_path / f"view_{idx:02d}"
                self.view_paths.append(view_path)
                if not view_path.exists():
                    view_path.mkdir(parents=True, exist_ok=True) 
                
                # DAPI at level 0 (for predictions)
                dapi_path = view_path / "dapi_level-0.tif"
                if not dapi_path.exists():
                    dapi = load_arr(
                        path, metadata=metadata, 
                        channels="DAPI", level=0, roi_idx=idx,
                        )
                    dapi = (norm_pct(dapi) * 255).astype("uint8")
                    io.imsave(dapi_path, dapi, check_contrast=False)
                    
                # Stack at process level 
                stack_path = view_path / f"stack_level-{self.process_level}.tif"
                if not stack_path.exists():
                    stack = load_arr(
                        path, metadata=metadata, 
                        channels="all", level=self.process_level, roi_idx=idx,
                        )
                    stack = ((stack + 8) / 16 ).astype("uint8") # 12bits to 8bits
                    io.imsave(stack_path, stack, check_contrast=False)                               
                        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):

        for p, path in enumerate(self.view_paths):
            dapi_path = path / "dapi_level-0.tif"
            msk_path  = path / f"mask_level-{self.process_level}.tif"

            if not msk_path.exists():
                print(f"process() : {path.parent.name}/{path.name}")
                
                if "unet" not in locals():
                    unet = UNet(
                        parameters=None, model_path=Path.cwd() / self.model_name)
                
                dapi = io.imread(dapi_path)
                dapi = dapi.astype("float32") / 255
            
                # Predict
                prd = unet.predict(dapi)
                
                # Get mask
                dscale_factor = 2 ** self.process_level
                prd = downscale_local_mean(prd, dscale_factor)
                msk_0 = prd > self.thresh_0
                msk_1 = prd > self.thresh_1
                msk = watershed(-prd, label(msk_1), mask=msk_0)
                msk[lbl_conn(msk, conn=2) == 2] = 0
                msk = remove_small_objects(
                    msk, min_size=0.5 * (2 ** self.process_level) ** 2)
                msk = ((msk > 0) * 255).astype("uint8")

                # Save mask & clear
                io.imsave(msk_path, msk, check_contrast=False)
                tf.keras.backend.clear_session()
                gc.collect()
            
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        for p, path in enumerate(self.view_paths):
            msk_path  = path / f"mask_level-{self.process_level}.tif"
            stack_path = path / f"stack_level-{self.process_level}.tif"
            results_path = path / f"intensities_level-{self.process_level}.csv"
            
            if not results_path.exists():
                print(f"analyse() : {path.parent.name}/{path.name}")
                
                msk = io.imread(msk_path) > 0
                stack = io.imread(stack_path)
                
                # Measure intensities
                msk = expand_labels(
                    label(msk), distance=1 * (2 ** self.process_level) ** 2)
                results = regionprops_table(
                    msk, intensity_image=np.moveaxis(stack, 0, -1), 
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
    stack = io.imread(stk_path)
    
    # -------------------------------------------------------------------------

    t0 = time.time()
    print("expand_labels() : ", end="", flush=False)
    
    msk = expand_labels(label(msk), distance=distance)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # t0 = time.time()
    # print("regionprops_table() : ", end="", flush=False)

    # results = regionprops_table(
    #     msk, intensity_image=np.moveaxis(stack, 0, -1), 
    #     properties=("label", "intensity_mean"),
    #     )
    # results.pop("label", None)
    # results = {
    #     channel : value 
    #     for channel, value in zip(channels, results.values())
    #     }
    # results = pd.DataFrame(results)
    # results.to_csv(Path.cwd() / "results.csv", index=False)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_labels(msk)