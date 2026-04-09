#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# bdtools
from bdtools.norm import norm_pct
from bdtools.unet.unet import UNet

# skimage
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, binary_closing, remove_small_objects
        
#%% function(s) : tools -------------------------------------------------------

def norm_12_to_8bits(arr):
    return ((arr + 8) / 16).astype("uint8")

def norm_pct_8bits(arr):
    if arr.ndim == 2:
        return (norm_pct(arr) * 255).astype("uint8")
    elif arr.ndim == 3:
        for i, img in enumerate(arr):
            arr[i] = (norm_pct(img) * 255).astype("uint8")
        return arr

#%% function(s) : load_metadata() ---------------------------------------------
        
def load_metadata(path, model_name=None, model=None):

    # Checks ------------------------------------------------------------------

    if (model_name is None) == (model is None):
        raise ValueError(
            "User must either provide 'model_name' or 'model', but not both"
            )    

    # Nested function(s) ------------------------------------------------------    

    def get_view_rois(path, metadata):
            
        # Make avg. preview
        preview = []
        with QPTiffFile(path) as qptiff:
            for channel in metadata["channels"]:
                preview.append(qptiff.read_region(
                    channel, level=metadata["level"][-1]
                    ))
        preview = np.mean(preview, axis=0)
        
        # Get mask
        sigma = 32 / metadata["pixsize"][-1]
        blurred = gaussian(preview, sigma=sigma)
        thresh = threshold_otsu(blurred)
        mask = blurred > thresh
        mask = remove_small_objects(mask, min_size=sigma * 1024)
        mask = binary_closing(mask, footprint=disk(sigma * 8))
        
        # Get views
        metadata["roi"] = []
        for props in regionprops(label(mask)):
            if props.area * (2 ** metadata["level"][-1]) ** 2  > 1e8:
                
                coords = props.coords
                y0 = np.min(coords[:, 0]) * 2 ** metadata["level"][-1]
                y1 = np.max(coords[:, 0]) * 2 ** metadata["level"][-1]
                x0 = np.min(coords[:, 1]) * 2 ** metadata["level"][-1]
                x1 = np.max(coords[:, 1]) * 2 ** metadata["level"][-1]
                
                roi = []
                for level in metadata["level"]:
                    pad = 512 / 2 ** level
                    y0s = (y0 / 2 ** level) - pad
                    y1s = (y1 / 2 ** level) + pad
                    x0s = (x0 / 2 ** level) - pad
                    x1s = (x1 / 2 ** level) + pad
                    roi.append((
                        int(np.maximum(y0s, 0)),
                        int(np.minimum(y1s, metadata["shape"][level][0])),
                        int(np.maximum(x0s, 0)),
                        int(np.minimum(x1s, metadata["shape"][level][1])),
                        ))
                    
                metadata["roi"].append(roi)

    # Execute -----------------------------------------------------------------

    metadata = {
        "channels" : [],
        "level"    : [],
        "pixsize"  : [],
        "shape"    : [],
        "preview"  : [],
        "mask"     : [],
        "stats"    : [],
        }
        
    # Fetch metadata
    with QPTiffFile(path) as qptiff:
        for level in range(len(qptiff.series[0].levels)):
            xml = ET.fromstring(qptiff.get_channel_info()[0]["raw_xml"])
            page = qptiff.series[0].levels[level].pages[0]
            height, width = page.shape
            pixsize = xml.find(".//PixelSizeMicrons")
            pixsize = float(pixsize.text) * 2 ** level
            metadata["channels"] = qptiff.get_biomarkers()
            metadata["level"   ].append(level)
            metadata["pixsize" ].append(pixsize)   
            metadata["shape"   ].append((height, width))
                
    # Get view ROIs
    get_view_rois(path, metadata)
        
    # Get preview(s)   
    for idx in range(len(metadata["roi"])):
        
        preview = load_arr(
            path, metadata=metadata, level=4, roi_idx=idx) # Hard coded level 4
        preview = np.stack(preview)
        metadata["preview"].append(preview)
        
        # Get mask(s) 
        if model is None:
            model = UNet(parameters=None, model_path=Path.cwd() / model_name)
        mask = model.predict(norm_pct(np.mean(preview, axis=0))) > 0.5
        metadata["mask"].append(mask)
        
        # Get statistics
        
        stats = {
            "avg" : [],
            "med" : [],
            "std" : [],
            "qtl" : [],
            }
        
        for c, channel in enumerate(metadata["channels"]):
            
            image = preview[c]
            stats["avg"].append(np.mean(image[mask]))
            stats["med"].append(np.median(image[mask]))
            stats["std"].append(np.std(image[mask]))
            
            quantiles = []
            for q, qtl in enumerate(np.arange(0.1, 1.0, 0.1)):
                quantiles.append(np.quantile(image[mask], qtl))
            stats["qtl"].append(quantiles)
        
        metadata["stats"].append(stats)
        
        
    return metadata

#%% function(s) : load_arr() --------------------------------------------------

def load_arr(path, metadata=None, channels="all", level=4, roi_idx=None):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_position():
        roi = metadata["roi"]
        return (
            roi[roi_idx][level][2], 
            roi[roi_idx][level][0],
            )  
    
    def get_shape():
        roi = metadata["roi"]
        return (
            roi[roi_idx][level][3] - roi[roi_idx][level][2],
            roi[roi_idx][level][1] - roi[roi_idx][level][0],
            )

    # Execute -----------------------------------------------------------------
    
    # Load metadata
    if metadata is None:
        metadata = load_metadata(path)
    
    # Get channels
    if channels == "all":
        channels = metadata["channels"]
    elif isinstance(channels, str):
        channels = [channels]
    
    # Load data
    arr = []
    with QPTiffFile(path) as qptiff:
        for channel in channels:
            if roi_idx is not None:
                arr.append(qptiff.read_region(
                    channel, level=level,
                    pos=get_position(),
                    shape=get_shape(),
                    ))
            else:
                arr.append(qptiff.read_region(
                    channel, level=level,
                    ))
                
    return np.stack(arr).squeeze()

#%% function(s) : crop --------------------------------------------------------

def get_crops(stk, msk, prp=None, crop_size=32):

    if prp is None:
        
        # Get centroids
        prp = regionprops_table(
            label(msk), properties=("label", "centroid"),
            )
        prp["y"] = prp.pop("centroid-0").astype("int32")
        prp["x"] = prp.pop("centroid-1").astype("int32")
        
    # Normalize (0 to 1)
    if np.issubdtype(stk.dtype, np.integer):
        stk = stk.astype("float32") / 255
    
    # Pad (acc. to crop half size)
    chs = crop_size // 2 # crop half size
    pad_width = ((0, 0), (chs, chs), (chs, chs))
    stk = np.pad(stk, pad_width=pad_width, mode="reflect")

    # Extract crops
    crops = []
    for (y, x) in zip(prp["y"], prp["x"]):
        y0 = y; y1 = y + crop_size
        x0 = x; x1 = x + crop_size
        if y1 - y0 != crop_size:
            print(y0, y1)
        if x1 - x0 != crop_size:
            print(x0, x1)
        crops.append(
            np.swapaxes(stk[..., y0:y1, x0:x1], 0, -1)
            )
        
    return np.stack(crops)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time
    
    # -------------------------------------------------------------------------
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    file_name = "251126_D7_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E2_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E6_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E7_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D1_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D2_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D3_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D4_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D5_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D6_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D8_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_E1_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_E3_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_E4_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_E5_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_E8_S2_Scan1_unmixed.qptiff"

    # -------------------------------------------------------------------------

    # Parameters
    path = data_path / file_name
    model_name = "model_256_binary_1024-43"
    channels = "all"
    level = 4
    idx = 0  
    
    # -------------------------------------------------------------------------
            
    t0 = time.time()
    print("load_metadata() : ", end="", flush=True)
    metadata = load_metadata(path, model_name=model_name)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     metadata["preview"][idx],
    #     gamma=0.3,
    #     )
    # vwr.add_image(
    #     metadata["mask"][idx],
    #     blending="additive", opacity=0.3, colormap="yellow",
    #     )
    
    # -------------------------------------------------------------------------
    
    # t0 = time.time()
    # print("load_arr() : ", end="", flush=True)
    # arr = load_arr(
    #     path, metadata=metadata, 
    #     channels=channels, level=level, roi_idx=idx,
    #     )
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(arr)
    
#%%

    idx = 0
    preview = metadata["preview"][idx]
    mask = metadata["mask"][idx]
    
    stats = {
        "avg" : [],
        "med" : [],
        "std" : [],
        "qtl" : [],
        }
    
    for c, channel in enumerate(metadata["channels"]):
        
        image = preview[c]
        stats["avg"].append(np.mean(image[mask]))
        stats["med"].append(np.median(image[mask]))
        stats["std"].append(np.std(image[mask]))
        
        # Quantiles
        quantiles = []
        for q, qtl in enumerate(np.arange(0.1, 1.0, 0.1)):
            quantiles.append(np.quantile(image[mask], qtl))
        stats["qtl"].append(quantiles)
    
     
    