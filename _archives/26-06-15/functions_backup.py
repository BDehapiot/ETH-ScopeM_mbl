#%% Imports -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    # Nested function(s) ------------------------------------------------------
    
    def mean_norm(df):
        dfn = df.copy()
        idx = df.filter(regex='|'.join(["entropy", "avg"])).columns
        dfn[idx] = df[idx] / df[idx].mean()
        idx = df.filter(regex='|'.join(["count"])).columns
        dfn[idx] = df[idx] / df[idx].sum()
        return dfn
    
    def class_avg(df):
        dfa = df.copy()
        dfa = df.groupby("class").mean(numeric_only=True)
        dfa = dfa.drop(columns=["idx"])
        dfa.insert(loc=0, column="count", value=df.groupby("class").size())
        dfa = dfa.reset_index()
        dfa.insert(loc=0, column="view", value=[vws_path.name] * n_classes)
        return dfa
    
    # Execute -----------------------------------------------------------------
    
    # Extrat data
    ncl_df = pd.DataFrame()
    n_nuclei = ncl_prd.shape[0]
    n_classes = len(classes)
    ncl_df["view"   ] = [vws_path.name] * n_nuclei
    ncl_df["idx"    ] = np.arange(n_nuclei)
    ncl_df["class"  ] = np.array(classes)[ncl_prd]
    ncl_df["entropy"] = ncl_ent
    for msr in ["avg"]:
        for chn in channels:
            key = f"{chn}_{msr}"
            ncl_df[key] = ncl_sts[msr][chn]
    ncl_dfa  = class_avg(ncl_df)
    ncl_dfn  = mean_norm(ncl_df)
    ncl_dfan = mean_norm(ncl_dfa)
        
    return ncl_df, ncl_dfa, ncl_dfn, ncl_dfan

#%% function(s) : Plots -------------------------------------------------------

def plot(df, dfa, dfn, dfan, channels, classes, colors):
    
    # Inputs ------------------------------------------------------------------
    
    plt.ioff()
    gcolor = "0.5"
    schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
    sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
    merged = df["view"].nunique(dropna=False) > 1
    
    # Nested function(s) ------------------------------------------------------
    
    def _ylim(ax, min_ymax=6):
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, np.maximum(min_ymax, ymax))
    
    def _plot(ax, vals, idx, color, merged=False):
                    
        # Boxplot
        ax.boxplot(
            vals, positions=[idx], widths=0.5, patch_artist=True,
            boxprops=dict(color=gcolor, facecolor="none", linewidth=1),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=gcolor, linewidth=1),
            capprops=dict(color=gcolor, linewidth=1),
            showfliers=False
            )
        
        # Mean line
        ax.axhline(y=1, color=gcolor, linestyle=":", linewidth=1)
        
        # x grid
        ax.grid(axis="x", linestyle=":", alpha=0.6, color=gcolor)
    
    # Execute -----------------------------------------------------------------
    
    # Initialize plot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 10))
    axes = axes.flatten()
    
    for cl, cla in enumerate(classes):

        ax = axes[cl]                   
        dfs = dfan if merged else dfn
        
        for ch, chn in enumerate(schannels):
            vals = np.array(dfs[dfs["class"] == cla][f"{chn}_avg"])
            _plot(ax, vals, ch, colors[cl], merged=merged)
                    
        # Legend
        _ylim(ax)
        ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
        ax.set_xticklabels(schannels, rotation=45, ha="center")
        ax.set_title(cla, pad=10)
    
    for i, measure in enumerate(["count", "entropy", "ASC_avg"]):
        
        ax = axes[i + 6] 
        
        if measure == "count":
            dfs = dfan
        if measure == "entropy":
            dfs = dfa if merged else df
        if measure == "ASC_avg":
            dfs = dfan if merged else dfn
        
        for cl, cla in enumerate(classes):
            vals = dfs[dfs["class"] == cla][measure]
            _plot(ax, vals, cl, colors[cl], merged=merged)
            
        # Legend
        if measure == "count":
            ax.set_ylim(0, 1)
            ax.set_ylabel("Count")
            ax.set_title("Relative count", pad=10)
        if measure == "entropy":
            ax.set_ylim(0, 1)
            ax.set_ylabel("Entropy")
            ax.set_title("Class pred. entropy", pad=10)
        if measure == "ASC_avg":
            _ylim(ax)
            ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
            ax.set_title("ASC Fluo. Int.", pad=10)
        ax.set_xticklabels(sclasses, rotation=45, ha="center")

    # Main title    
    fig.suptitle(
        "Merged views" if merged else dfa["view"].iloc[0],
        fontsize=16, fontweight="bold", color="k",
        y=0.95, x=0.025, ha="left",
        )

    # Secondary text
    text = (
        f"n views = {dfa['view'].nunique()}\n"
        f"n nuclei = {dfa['count'].sum():,}\n"
        f"avg. entropy = {dfa['entropy'].mean():.3f}"
        )
    fig.text(
        0.05, 0.92, text,
        fontsize=12, color=gcolor,
        ha="left", va="top",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.close(fig) 
    plt.ion()

    return fig