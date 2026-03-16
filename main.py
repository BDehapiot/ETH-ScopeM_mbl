#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# skimage
from skimage.measure import label, regionprops

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_mbl\data")
# file_name = "251126_D7_S2_Scan1.unmixed.qptiff"
# file_name = "251126_E2_S2_Scan1.unmixed.qptiff"
# file_name = "251126_E6_S2_Scan1.unmixed.qptiff"
file_name = "251126_E7_S2_Scan1.unmixed.qptiff"

#%% Function(s) ---------------------------------------------------------------

def load_metadata(file_path):
    metadata = {
        "levels"   : [],
        "shapes"   : [],
        "channels" : [],
        "pixsize"  : [],
        "rois"    : [],
        }
    with QPTiffFile(file_path) as qptiff:
        metadata["channels"] = qptiff.get_biomarkers()
        for level in range(len(qptiff.series[0].levels)):
            page = qptiff.series[0].levels[level].pages[0]
            height, width = page.shape
            xml = ET.fromstring(qptiff.get_channel_info()[0]["raw_xml"])
            pixsize = xml.find('.//PixelSizeMicrons')
            pixsize = float(pixsize.text) * 2 ** level
            metadata["levels"].append(level)
            metadata["shapes"].append((height, width))
            metadata["pixsize"].append(pixsize)      
    metadata["rois"] = get_rois(file_path)
    return metadata

def load_data(file_path, channels="all", level=5, roi=None):
    
    def get_roi_pos(roi, level):
        return (roi[level][0], roi[level][2])
    def get_roi_shape(roi, level):
        return (roi[level][1] - roi[level][0], roi[level][3] - roi[level][2])
    
    data = []
    with QPTiffFile(file_path) as qptiff:
        
        if channels == "all":
            channels = qptiff.get_biomarkers()
        elif isinstance(channels, str):
            channels = [channels]
            
        for channel in channels:
            if roi is None:
                data.append(qptiff.read_region(
                    channel, level=level
                    ))
            else:
                data.append(qptiff.read_region(
                    channel, level=level,
                    pos=get_roi_pos(roi, level),
                    shape=get_roi_shape(roi, level),
                    ))
                
    return np.stack(data)

def get_rois(file_path):
    rois = []
    data = load_data(file_path, channels="all", level=5, roi=None)
    data_avg = np.mean(data, axis=0)
    for props in regionprops(label(data_avg > 0)):
        roi = []
        coords = props.coords
        y0, y1 = np.min(coords[:, 0]), np.max(coords[:, 0])
        x0, x1 = np.min(coords[:, 1]), np.max(coords[:, 1])
        for l, lvl in enumerate(range(level, -1, -1)):
            roi.append((
                y0 * 2 ** lvl, 
                y1 * 2 ** lvl, 
                x0 * 2 ** lvl, 
                x1 * 2 ** lvl,
                ))
        rois.append(roi)
    return rois

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time

    # Parameters
    file_path = data_path / file_name
    channels = "all"
    level = 5
    roi = None
    
    # -------------------------------------------------------------------------
        
    t0 = time.time()
    print("load_metadata() : ", end="", flush=True)
    metadata = load_metadata(file_path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    t0 = time.time()
    print("load_data() : ", end="", flush=True)
    data = load_data(file_path, channels=channels, level=level, roi=roi)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # # get_view_rois()
    # rois = get_view_rois(file_path)
    
    # # load_data()
    # data_crop = load_data(file_path, channels="DAPI", level=0, roi=rois[0])

    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(data_crop)