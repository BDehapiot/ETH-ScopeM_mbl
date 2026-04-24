#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# bdtools
from bdtools.conn import lbl_conn
from bdtools.norm import norm_pct

# skimage
from skimage.segmentation import watershed
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, binary_closing, remove_small_objects
        
#%% function(s) : normalization -----------------------------------------------

def norm_12_to_8bits(arr):
    if arr.ndim == 2:
        return ((arr + 8) / 16).astype("uint8")
    elif arr.ndim == 3:
        for i, img in enumerate(arr):
            arr[i] = ((img + 8) / 16)
        return arr.astype("uint8")
            
def norm_pct_8bits(arr):
    if arr.ndim == 2:
        return (norm_pct(arr) * 255).astype("uint8")
    elif arr.ndim == 3:
        for i, img in enumerate(arr):
            arr[i] = (norm_pct(img) * 255).astype("uint8")
        return arr   
    
def prep_preview(pvw, channels=[2, 5, 6]):
    return np.mean(
        norm_pct(pvw)[channels, ...], axis=0).astype("float32") 

def norm_data(arr, its, channel_idx=None):
    
    if arr.ndim == 2 and channel_idx is None:
        raise ValueError("please provide 'channel_idx' as int for 2D ndarray")   
    
    arr = arr.astype("float32")
    
    if arr.ndim == 2:
        avg = its["avg"][channel_idx]
        std = its["std"][channel_idx]
        arr -= avg
        arr /= std
        arr = norm_pct(arr)
        
    elif arr.ndim == 3:
        for channel_idx in range(arr.shape[0]):
            avg = its["avg"][channel_idx]
            std = its["std"][channel_idx]
            arr[channel_idx] -= avg
            arr[channel_idx] /= std
            arr[channel_idx] = norm_pct(arr[channel_idx])
    
    return arr

#%% function(s) : save compressed ---------------------------------------------

def save_compressed_mask(arr, path):
    arr_bin = np.packbits(arr > 0)        
    np.savez_compressed(path, data=arr_bin, org_shape=arr.shape)

def load_compressed_mask(path):
    with np.load(path) as loader:
        arr_bin = loader["data"]
        shape = loader["org_shape"]
        unpacked = np.unpackbits(arr_bin)[:np.prod(shape)]
        return unpacked.reshape(shape).astype(np.uint8) * 255
    
def save_compressed_stack(arr, path):
    np.savez_compressed(path, data=arr)

def load_compressed_stack(path):
    with np.load(path) as loader:
        return loader["data"]

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

def load_view(path, metadata=None, channels="all", level=4, roi_idx=None):
    
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

#%% function(s) : get_intensities() -------------------------------------------

def get_intensities(pvw, msk):
    msk = msk > 0
    its = {"avg" : [], "med" : [], "std" : [], "qtl" : []}
    avg, med, std, qtl = [], [], [], []
    for channel in range(pvw.shape[0]):
        qt = []
        avg.append(np.mean(pvw[channel][msk]))
        med.append(np.median(pvw[channel][msk]))
        std.append(np.std(pvw[channel][msk]))            
        for q in np.arange(0.1, 1.0, 0.1):
            qt.append(np.quantile(pvw[channel][msk], q))
        qtl.append(qt)
    its["avg"] = np.stack(avg)
    its["med"] = np.stack(med)
    its["std"] = np.stack(std)
    its["qtl"] = np.stack(qtl)
    return its

#%% function(s) : nuclei ------------------------------------------------------
    
def get_nuclei_mask(prd, thresh_0=0.25, thresh_1=0.50, prc_level=1):
    msk_0 = prd > thresh_0
    msk_1 = prd > thresh_1
    msk = watershed(-prd, label(msk_1), mask=msk_0)
    msk[lbl_conn(msk, conn=2) == 2] = 0
    msk = remove_small_objects(
        msk, min_size=0.5 * (2 ** prc_level) ** 2)
    return msk > 0

def get_nuclei_centroids(msk, n_levels=5):
    props = regionprops_table(label(msk), properties=("label", "centroid"))
    y, x = props["centroid-0"], props["centroid-1"]
    ctd = {"y" : [], "x" : []}
    for level in range(n_levels):
        ctd["y"].append(y / 2 ** level) 
        ctd["x"].append(x / 2 ** level) 
    return ctd

#%% function(s) : get_crops() -------------------------------------------------

def get_crops(stk, ctd, crop_size=32, crop_fraction=0.1, stk_level=1):
    
    # Pad (acc. to crop half size)
    chs = crop_size // 2 # crop half size
    pad_width = ((0, 0), (chs, chs), (chs, chs))
    stk = np.pad(stk, pad_width=pad_width, mode="reflect")

    # Extract coordinates
    ys = ctd["y"][stk_level]
    xs = ctd["x"][stk_level]
    
    # Select random coordinates
    crop_idx = np.random.choice(
        np.arange(0, len(ys)), 
        size=int(len(ys) * crop_fraction), 
        replace=False,
        )
    ys = ys[crop_idx].astype(int)
    xs = xs[crop_idx].astype(int)
    
    # Extract crops
    crops = []
    for (y, x) in zip(ys, xs):
        y0 = y; y1 = y + crop_size
        x0 = x; x1 = x + crop_size
        if y1 - y0 != crop_size: print(y0, y1)
        if x1 - x0 != crop_size: print(x0, x1)
        crops.append(stk[..., y0:y1, x0:x1])
        
    return np.stack(crops)

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
