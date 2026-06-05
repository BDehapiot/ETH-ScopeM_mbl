#%% Imports -------------------------------------------------------------------

import numpy as np
import pandas as pd
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
       
#%% function(s) : compression -------------------------------------------------

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
    
def prep_preview(pvw_raw, channels=[2, 5, 6]):
    return np.mean(
        norm_pct(pvw_raw)[channels, ...], axis=0).astype("float32") 

def norm_data(arr, pvw_sts, channel=None):
    
    if arr.ndim == 2 and channel is None:
        raise ValueError("please provide 'channel' as str for 2D ndarray")   
    
    arr = arr.astype("float32")
    
    if arr.ndim == 2:
        avg = pvw_sts["avg"][channel]
        std = pvw_sts["std"][channel]
        arr -= avg
        arr /= std
        arr = norm_pct(arr)
        
    elif arr.ndim == 3:
        channels = list(pvw_sts["avg"])
        for c, chn in enumerate(channels):
            avg = pvw_sts["avg"][chn]
            std = pvw_sts["std"][chn]
            arr[c] -= avg
            arr[c] /= std
            arr[c] = norm_pct(arr[c])
    
    return arr

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

    
#%% function(s) : nuclei segmentation -----------------------------------------
    
def get_nuclei_mask(ncl_prd, thresh_0=0.25, thresh_1=0.50, prc_level=1):
    msk_0 = ncl_prd > thresh_0
    msk_1 = ncl_prd > thresh_1
    msk = watershed(-ncl_prd, label(msk_1), mask=msk_0)
    msk[lbl_conn(msk, conn=2) == 2] = 0
    msk = remove_small_objects(
        msk, min_size=0.5 * (2 ** prc_level) ** 2)
    return msk > 0

def get_nuclei_centroids(ncl_msk, n_levels=5):
    props = regionprops_table(label(ncl_msk), properties=("label", "centroid"))
    y, x = props["centroid-0"], props["centroid-1"]
    ctd = {"y" : [], "x" : []}
    for level in range(n_levels):
        ctd["y"].append(y / 2 ** level) 
        ctd["x"].append(x / 2 ** level) 
    return ctd

#%% function(s) : statistics --------------------------------------------------

def get_preview_stats(pvw_raw, pvw_msk, channels):
    pvw_msk = pvw_msk > 0
    pvw_sts = {"avg" : {}, "std" : {}}
    for c, chn in enumerate(channels):
        pvw_sts["avg"][chn] = np.mean(pvw_raw[c, ...][pvw_msk])
        pvw_sts["std"][chn] = np.mean(pvw_raw[c, ...][pvw_msk])
    return pvw_sts

def format_preview_stats():
    pass

def get_nuclei_stats(crp, channels):        
    ncl_sts = {"avg" : {}, "std" : {}}
    for c, chn in enumerate(channels):
        ncl_sts["avg"][chn] = np.mean(crp[:, c, ...], axis=(1, 2))
        ncl_sts["std"][chn] = np.std( crp[:, c, ...], axis=(1, 2))
    return ncl_sts

def format_nuclei_stats(
        ncl_prd, ncl_ent, ncl_sts, vws_path, 
        channels, classes
        ):
    
    # Extrat data
    ncl_df = pd.DataFrame()
    n_nuclei = ncl_prd.shape[0]
    n_classes = len(classes)
    ncl_df["vws"    ] = [vws_path.name] * n_nuclei
    ncl_df["idx"    ] = np.arange(n_nuclei)
    ncl_df["class"  ] = np.array(classes)[ncl_prd]
    ncl_df["entropy"] = ncl_ent
    for msr in ["avg"]:
        for chn in channels:
            key = f"{chn}_{msr}"
            ncl_df[key] = ncl_sts[msr][chn]
    
    # Class avg.
    ncl_df_avg = ncl_df.groupby("class").mean(numeric_only=True)
    ncl_df_avg = ncl_df_avg.drop(columns=["idx"])
    ncl_df_avg.insert(
        loc=0, column="count", value=ncl_df.groupby("class").size())
    ncl_df_avg.insert(
        loc=0, column="class", value=classes)
    ncl_df_avg.insert(
        loc=0, column="vws", value=[vws_path.name] * n_classes)
    
    # Mean norm. of class avg.
    ncl_df_avg_nrm = ncl_df_avg.copy()
    idx = ncl_df_avg_nrm.filter(regex='|'.join(["entropy", "avg"])).columns
    ncl_df_avg_nrm[idx] = ncl_df_avg_nrm[idx] / ncl_df_avg_nrm[idx].mean()
    
    return ncl_df, ncl_df_avg, ncl_df_avg_nrm

#%% function(s) : crops -------------------------------------------------------

def get_crops(stk, ncl_ctd, crop_size=32, crop_fraction=None, stk_level=1):
    
    # Pad (acc. to crop half size)
    chs = crop_size // 2 # crop half size
    pad_width = ((0, 0), (chs, chs), (chs, chs))
    stk = np.pad(stk, pad_width=pad_width, mode="reflect")

    # Extract coordinates
    ys = ncl_ctd["y"][stk_level].astype(int)
    xs = ncl_ctd["x"][stk_level].astype(int)
    
    # Select random coordinates
    if crop_fraction is not None:
        crop_idx = np.random.choice(
            np.arange(0, len(ys)), 
            size=int(len(ys) * crop_fraction), 
            replace=False,
            )
        ys = ys[crop_idx]
        xs = xs[crop_idx]
    
    # Extract crops
    crops = []
    for (y, x) in zip(ys, xs):
        y0 = y; y1 = y + crop_size
        x0 = x; x1 = x + crop_size
        if y1 - y0 != crop_size: print(y0, y1)
        if x1 - x0 != crop_size: print(x0, x1)
        crops.append(stk[..., y0:y1, x0:x1])
        
    return np.stack(crops)

def prepare_crops(crp):
    crp = np.delete(crp, (0, 4, -1), axis=1) # remove DAPI, ASC & autofluo channels
    crp = np.transpose(crp, (0, 2, 3, 1)) # transposition
    crp = crp.astype("float32") / 255 # normalize
    return crp

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
    roi_idx = 0
    
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
        channels=channels, level=level, roi_idx=roi_idx,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(vws, gamma=0.5)
