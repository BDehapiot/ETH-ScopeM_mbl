#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# bdtools
from bdtools.norm import norm_gcn, norm_pct

# skimage
from skimage.transform import rescale
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, binary_closing, remove_small_objects
        
#%% function(s) : normalization -----------------------------------------------

def norm_12_to_8bits(arr):
    if arr.ndim == 2:
        return ((arr + 8) / 16).astype("uint8")
    elif arr.ndim == 3:
        for i, img in enumerate(arr):
            arr[i] = ((img + 8) / 16).astype("uint8")
        return arr  
            
def norm_pct_8bits(arr):
    if arr.ndim == 2:
        return (norm_pct(arr) * 255).astype("uint8")
    elif arr.ndim == 3:
        for i, img in enumerate(arr):
            arr[i] = (norm_pct(img) * 255).astype("uint8")
        return arr   
    
def norm_preview(pvw, channels=[2, 5, 6], dtype="uint8"):
    if dtype == "uint8":
        return np.mean(
            norm_pct_8bits(pvw)[channels, ...], axis=0).astype("uint8")  
    elif dtype == "float32":
        return np.mean(
            norm_pct(pvw)[channels, ...], axis=0).astype("float32") 

def norm_dapi(dap, msk, dap_level=1, pvw_level=4, dtype="uint8"):
    msk = msk > 0
    rfactor = 2 ** (pvw_level - dap_level)
    msk = rescale(msk, rfactor, order=0)     
    msk = norm_pct(norm_gcn(dap, mask=msk), mask=msk)
    if dtype == "uint8":
        return (msk * 255).astype("uint8")
    elif dtype == "float32":
        return msk.astype("float32")

#%% function(s) : get_slide_metadata() ----------------------------------------
        
def get_slide_metadata(path):

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
            pixsize = xml.find(".//PixelSizeMicrons")
            pixsize = float(pixsize.text) * 2 ** level
            metadata["channels"] = qptiff.get_biomarkers()
            metadata["level"   ].append(level)
            metadata["pixsize" ].append(pixsize)   
            metadata["shape"   ].append((height, width))
                
    # Get view ROIs
    get_view_rois(path, metadata)
        
    return metadata

#%% function(s) : load_view() -------------------------------------------------

def load_view(path, metadata=None, channels="all", level=4, view_idx=None):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_position():
        roi = metadata["roi"]
        return (
            roi[view_idx][level][2], 
            roi[view_idx][level][0],
            )  
    
    def get_shape():
        roi = metadata["roi"]
        return (
            roi[view_idx][level][3] - roi[view_idx][level][2],
            roi[view_idx][level][1] - roi[view_idx][level][0],
            )

    # Execute -----------------------------------------------------------------
    
    # Load metadata
    if metadata is None:
        metadata = get_slide_metadata(path)
    
    # Get channels
    if channels == "all":
        channels = metadata["channels"]
    elif isinstance(channels, str):
        channels = [channels]
    
    # Load data
    arr = []
    with QPTiffFile(path) as qptiff:
        for channel in channels:
            if view_idx is not None:
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

#%% function(s) : get_statistics() --------------------------------------------

def get_statistics(pvw, msk):
    msk = msk > 0
    sts = {"avg" : [], "med" : [], "std" : [], "qtl" : []}
    avg, med, std, qtl = [], [], [], []
    for channel in range(pvw.shape[0]):
        qt = []
        avg.append(np.mean(pvw[channel][msk]))
        med.append(np.median(pvw[channel][msk]))
        std.append(np.std(pvw[channel][msk]))            
        for q in np.arange(0.1, 1.0, 0.1):
            qt.append(np.quantile(pvw[channel][msk], q))
        qtl.append(qt)
    sts["avg"] = np.stack(avg)
    sts["med"] = np.stack(med)
    sts["std"] = np.stack(std)
    sts["qtl"] = np.stack(qtl)
    return sts

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
    file_name = "251126_E7_S2_Scan1_unmixed.qptiff"
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
    level = 4
    view_idx = 0
    
    # -------------------------------------------------------------------------
            
    t0 = time.time()
    print("get_slide_metadata() : ", end="", flush=True)
    mtd = get_slide_metadata(path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("load_view() : ", end="", flush=True)
    vws = load_view(
        path, metadata=mtd, 
        channels=channels, level=level, view_idx=view_idx
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(vws, gamma=0.5)
