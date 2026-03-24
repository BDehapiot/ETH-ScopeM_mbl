#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# skimage
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, binary_closing, remove_small_objects
        
#%% function(s) : load metadata & data ----------------------------------------
        
def load_metadata(path):

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
        }
    
    # Fetch metadata
    with QPTiffFile(path) as qptiff:
        for level in range(len(qptiff.series[0].levels)):
            xml = ET.fromstring(qptiff.get_channel_info()[0]["raw_xml"])
            page = qptiff.series[0].levels[level].pages[0]
            height, width = page.shape
            pixsize = xml.find('.//PixelSizeMicrons')
            pixsize = float(pixsize.text) * 2 ** level
            metadata["channels"] = qptiff.get_biomarkers()
            metadata["level"   ].append(level)
            metadata["pixsize" ].append(pixsize)   
            metadata["shape"   ].append((height, width))
                
    # Get view ROIs
    get_view_rois(path, metadata)
    
    return metadata

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
    channels = "all"
    level = 3
    idx = 0   
    
    # -------------------------------------------------------------------------
        
    t0 = time.time()
    print("load_metadata() : ", end="", flush=True)
    metadata = load_metadata(path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(preview)
    # vwr.add_image(mask, blending="additive", opacity=0.25, colormap="yellow")
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("load_arr() : ", end="", flush=True)
    arr = load_arr(
        path, metadata=metadata, 
        channels=channels, level=level, roi_idx=idx,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(arr)