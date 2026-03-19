#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# skimage
from skimage.measure import label, regionprops
        
#%% function(s) : load metadata & data ----------------------------------------
        
def load_metadata(path):
    
    metadata = {
        "levels"    : [],
        "height"    : [],
        "width"     : [],
        "channels"  : [],
        "pixsizes"  : [],
        "positions" : [],
        "shapes"    : [],
        }
    
    # Fetch metadata
    with QPTiffFile(path) as qptiff:
        for level in range(len(qptiff.series[0].levels)):
            xml = ET.fromstring(qptiff.get_channel_info()[0]["raw_xml"])
            page = qptiff.series[0].levels[level].pages[0]
            height, width = page.shape
            pixsize = xml.find('.//PixelSizeMicrons')
            pixsize = float(pixsize.text) * 2 ** level
            metadata["levels"  ].append(level)
            metadata["height"  ].append(height)
            metadata["width"   ].append(width)
            metadata["channels"] = qptiff.get_biomarkers()
            metadata["pixsizes"].append(pixsize)    
    
    # Make avg. preview
    preview = []
    with QPTiffFile(path) as qptiff:
        for channel in metadata["channels"]:
            preview.append(qptiff.read_region(
                channel, level=metadata["levels"][-1]
                ))
    preview = np.mean(preview, axis=0)
    
    # Get view coordinates
    for props in regionprops(label(preview > 0)):
        coords = props.coords
        y0, y1 = np.min(coords[:, 0]), np.max(coords[:, 0])
        x0, x1 = np.min(coords[:, 1]), np.max(coords[:, 1])
        position, shape = [], []
        for l, lvl in enumerate(range(metadata["levels"][-1], -1, -1)):
            y0s = y0 * 2 ** lvl
            y1s = np.minimum(
                y1 * 2 ** lvl, metadata["height"][l])
            x0s = x0 * 2 ** lvl
            x1s = np.minimum(
                x1 * 2 ** lvl, metadata["width"][l])
            position.append((x0s, y0s))
            shape.append((x1s - x0s, y1s - y0s))
        metadata["positions"].append(position)
        metadata["shapes"   ].append(shape)
            
    return metadata

def load_arr(path, metadata=None, channels="all", level=5, view=0):
    
    # Load metadata (if not provided)
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
            arr.append(qptiff.read_region(
                channel, level=level,
                pos=metadata["positions"][view][level],
                shape=metadata["shapes"][view][level],
                ))
                
    return np.stack(arr)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time
    
    # -------------------------------------------------------------------------
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    # file_name = "251126_D7_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E2_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E6_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E7_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D1_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D2_S2_Scan1_unmixed.qptiff"
    # file_name = "251127_D3_S2_Scan1_unmixed.qptiff"
    file_name = "251127_D4_S2_Scan1_unmixed.qptiff"
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
    level = 4
    view = 0
    
    # -------------------------------------------------------------------------
        
    t0 = time.time()
    print("load_metadata() : ", end="", flush=True)
    metadata = load_metadata(path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    t0 = time.time()
    print("load_arr() : ", end="", flush=True)
    arr = load_arr(
        path, metadata=None, channels=channels, level=level, view=view)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(arr)