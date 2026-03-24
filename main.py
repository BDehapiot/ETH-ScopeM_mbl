#%% Imports -------------------------------------------------------------------

import gc
import pickle
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import load_metadata, load_arr

# bdtools
from bdtools.norm import norm_pct
from bdtools.conn import lbl_conn
from bdtools.unet.unet import UNet

# skimage
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_objects

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Paths
    "data_path"     : Path("D:\local_mbl\data"),
    
    # extract()
    "preview_level" : 4,
    
    # process()
    "model_name"    : "model_256_edt_512-28",
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
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.output_paths = [
            self.data_path / f"{path.stem[:12]}"
            for path in self.slide_paths
            ]
        self.metadata_paths = [
            path / "metadata.pkl"
            for path in self.output_paths
            ]
        self.preview_paths = [
            path / "preview.tif"
            for path in self.output_paths
            ]
        
        # Setup output directory
        for output_path in self.output_paths:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True) 

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        self.dapi_paths = []
        for p, path in enumerate(self.slide_paths): 
            output_path   = self.output_paths[p]
            metadata_path = self.metadata_paths[p]
            preview_path  = self.preview_paths[p]
            
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

            # DAPI
            for idx in range(len(metadata["roi"])):
                dapi_path = output_path / f"dapi_{idx:02d}.tif"
                self.dapi_paths.append(dapi_path)
                if not dapi_path.exists():
                    dapi = load_arr(
                        path, metadata=metadata, 
                        channels="DAPI", level=0, roi_idx=idx,
                        )
                    dapi = (norm_pct(dapi) * 255).astype("uint8")
                    io.imsave(dapi_path, dapi, check_contrast=False)
                        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):
        
        unet = UNet(
            parameters=None, model_path=Path.cwd() / self.model_name)
        
        self.msk_paths = [
            path.parent / (str(path.stem) + "_msk.tif") 
            for path in self.dapi_paths
            ]
        
        for p, path in enumerate(self.dapi_paths):
            msk_path = self.msk_paths[p]

            # DAPI mask
            if p == 0:
            # if not msk_path.exists():
                print(f"process() : {path.parent.name}/{path.name}")
                dapi = io.imread(path)
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
                
                self.msk = msk
                
                # Save mask & clear
                io.imsave(msk_path, msk, check_contrast=False)
                tf.keras.backend.clear_session()
                gc.collect()
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    msk = main.msk
    
#%% 
    
    import time
    from skimage.measure import regionprops
    from skimage.segmentation import expand_labels
    
    lbl = label(msk)

    t0 = time.time()
    print("expand_labels() : ", end="", flush=False)
    lbl = expand_labels(lbl, distance=5)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    area = []
    for props in regionprops(lbl):
        area.append(props.area)

    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_labels(lbl)
        
#%% 

    # import time

    # # -------------------------------------------------------------------------

    # # Parameters
    # idx = 5
    # dapi_path = main.dapi_paths[idx]
    # metadata_path = dapi_path.parent / "metadata.pkl" 
    # with open(metadata_path, "rb") as file:
    #     metadata = pickle.load(file)
    
    # # Load & normalize dapi
    # dapi = io.imread(dapi_path)
    # dapi = dapi.astype("float32") / 255

    # t0 = time.time()
    # print("process() : ", end="", flush=False)
    # msk = main.process(dapi)
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(msk)
