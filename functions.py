#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qptifffile import QPTiffFile
import xml.etree.ElementTree as ET

# bdtools
from bdtools.conn import lbl_conn
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.segmentation import watershed
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, binary_closing, remove_small_objects

# scipy
from scipy.ndimage import distance_transform_edt
       
#%% function(s) : save & load -------------------------------------------------

def imread_channel(path, channel_idx=0, normalize=True):
    if isinstance(channel_idx, int):
        channel_idx = [channel_idx]
    img_memmap = tifffile.memmap(path)
    img = img_memmap[channel_idx, :, :].squeeze()
    if normalize:
        return img.astype("float32") / 255
    else:
        return img

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

#%% function(s) : normalization & scaling -------------------------------------

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
        avgs = np.array([pvw_sts["avg"][chn] for chn in channels])
        stds = np.array([pvw_sts["std"][chn] for chn in channels])
        avgs = avgs[:, np.newaxis, np.newaxis]
        stds = stds[:, np.newaxis, np.newaxis]
        arr -= avgs
        arr /= stds
        for c, chn in enumerate(channels):
            arr[c] = norm_pct(arr[c], sample_fraction=0.1)
    
    return arr

def get_sfactor(lvl_in, lvl_out, ndim=1, pixel_size=None):
    if pixel_size is None: pixel_size = 1.0
    return ((2 ** (lvl_in - lvl_out)) * pixel_size) ** ndim

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

#%% function(s) : preview -----------------------------------------------------

def get_preview_processed(pvw_raw, channels=[2, 5, 6]):
    pvw_prc = np.mean(
        norm_pct(pvw_raw)[channels, ...], axis=0).astype("float32") 
    return pvw_prc

def get_preview_stats(pvw_raw, pvw_msk, channels):
    pvw_msk = pvw_msk > 0
    pvw_sts = {"avg" : {}, "std" : {}}
    for c, chn in enumerate(channels):
        pvw_sts["avg"][chn] = np.mean(pvw_raw[c, ...][pvw_msk])
        pvw_sts["std"][chn] = np.mean(pvw_raw[c, ...][pvw_msk])
    return pvw_sts

#%% function(s) : get_obj_mask() ----------------------------------------------

def get_obj_mask(
        prd, pvw_msk, thresh_0=0.25, thresh_1=0.50, min_size=8, max_size=None, 
        prd_level=0, msk_level=1, pvw_level=4,
        ):
    
    # Nested function(s) ------------------------------------------------------
    
    def filter_object_size(msk, min_size=8, max_size=None):
        if min_size is not None:
            msk = remove_small_objects(msk, min_size=min_size)
        if max_size is not None:
            msk_large = remove_small_objects(msk, min_size=max_size + 1)
            msk = msk & ~msk_large
        return msk
    
    # Execute -----------------------------------------------------------------
    
    # Initialize
    rf_prd = get_sfactor(prd_level, msk_level, ndim=1, pixel_size=None)
    rf_pvw = get_sfactor(pvw_level, msk_level, ndim=1, pixel_size=None)
    if min_size is not None:
        min_size = min_size * (rf_prd ** 2) 
    if max_size is not None:
        max_size = max_size * (rf_prd ** 2) 
    
    # Rescale
    if rf_prd != 1: 
        prd = rescale(prd, rf_prd)
    if rf_pvw != 1:
        pvw_msk = rescale(pvw_msk, rf_pvw, order=0)
        
    # Get mask
    if thresh_1 is not None:
        msk_0 = prd > thresh_0
        msk_1 = prd > thresh_1
        msk = watershed(-prd, label(msk_1), mask=msk_0)
        msk[lbl_conn(msk, conn=2) == 2] = 0
        msk = msk > 0
    else:
        msk = prd > thresh_0
    
    # Remove "out of" pvw_msk objects
    msk[pvw_msk == 0] = 0
    
    # Filter object size
    msk = filter_object_size(msk, min_size=min_size, max_size=max_size)
        
    return msk

def get_mask_from_coords(coords, labels=None, shape=(2048, 2048)):
    if labels is None:
        msk = np.zeros(shape, dtype=bool)
        for coord in coords:
            msk[tuple(coord.T)] = True
    else:
        msk = np.zeros(shape, dtype="uint8")
        for c, coord in enumerate(coords):
            msk[tuple(coord.T)] = labels[c]
    return msk

#%% function(s) : get_obj_properties() ----------------------------------------

def get_obj_properties(nrm, msk, stk_level=1, pixel_size=0.5):

    sfactor = get_sfactor(stk_level, 0, ndim=2, pixel_size=pixel_size)
    
    props = regionprops_table(
        label(msk), intensity_image=nrm, 
        properties=(
            "centroid", "area", 
            "coords", "eccentricity",
            "intensity_max", "intensity_min",
            "intensity_mean", "intensity_std",
            ),
        )

    return {
        "y"      : props["centroid-0"].astype("uint16"),  
        "x"      : props["centroid-1"].astype("uint16"),  
        "area"   : (props["area"] * sfactor).astype("uint16") , 
        "coords" : [coord.astype("uint16") for coord in props["coords"]], 
        "eccentricity"   : props["eccentricity"].astype("float32"), 
        "intensity_max"  : props["intensity_max"].astype("float32"),  
        "intensity_min"  : props["intensity_min"].astype("float32"), 
        "intensity_mean" : props["intensity_mean"].astype("float32"), 
        "intensity_std"  : props["intensity_std"].astype("float32"), 
        }  

#%% function(s) : get_obj_crops() ---------------------------------------------

def get_obj_crops(stk, ys, xs, crop_size=32, crop_fraction=None):

    # Check inputs
    ndim = stk.ndim
    
    # Pad (acc. to crop half size)
    hlf = crop_size // 2 # crop half size
    if ndim == 3:
        pad_width = ((0, 0), (hlf, hlf), (hlf, hlf))
    elif ndim == 2:
        pad_width = ((hlf, hlf), (hlf, hlf))
    stk = np.pad(stk, pad_width=pad_width, mode="reflect")
        
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
        if ndim == 3:
            crops.append(stk[..., y0:y1, x0:x1])
        elif ndim == 2:
            crops.append(stk[y0:y1, x0:x1])
            
    # Format output
    if ndim == 3:
        return np.transpose(np.stack(crops), (0, 2, 3, 1))
    else:
        return np.stack(crops)
    
#%% function(s) : get_ncl_intensities() ---------------------------------------

def get_ncl_intensities(
        stk_nrm, ncl_prp, asc_msk, channels, 
        crop_size=32, normalize=True, 
        stk_level=1, pixel_size=0.5
        ):
            
    # Initialize
    
    ncl_int = {
        "y" : ncl_prp["y"], 
        "x" : ncl_prp["x"],
        } 
    
    channels = channels.copy()
    if "ASC" in channels:
        channels.remove("ASC")
    channels.extend(["asc_all", "asc_spk", "asc_nsp"])
    
    # Process ASC channel
    asc_nrm = stk_nrm[4, ...]
    stk_nrm = np.delete(stk_nrm, obj=4, axis=0)
    stk_nrm = np.concatenate((
        stk_nrm, np.stack([
            asc_nrm * ((asc_msk > 0) & (asc_msk < 3)),
            asc_nrm * (asc_msk == 1),
            asc_nrm * (asc_msk == 2),
            ])), axis=0)
    
    # Get crops
    sfactor = get_sfactor(0, stk_level, ndim=1, pixel_size=pixel_size)
    ncl_crp = get_obj_crops(
        stk_nrm, ncl_prp["y"], ncl_prp["x"], 
        crop_size=int(crop_size * sfactor)
        )
    
    # Measure intensities
    for c, chn in enumerate(channels):
        _ncl_int = np.mean(ncl_crp[..., c], axis=(1, 2))
        if normalize:
            _ncl_int /= np.mean(_ncl_int) 
        ncl_int[f"{chn}_int"] = _ncl_int
        
    return ncl_int

#%% function(s) : nuclei distances (to asc) -----------------------------------

def get_ncl_distances(asc_msk, ncl_prp, stk_level=1, pixel_size=0.5):
            
    def _get_ncl_distances(msk, sfactor):
        asc_edt = distance_transform_edt(~msk)
        ncl_dst = asc_edt[(
            ncl_prp["y"].astype(int), 
            ncl_prp["x"].astype(int),
            )]
        return ncl_dst.astype("float32")
    
    # Initialize
    ncl_dst = {
        "y" : ncl_prp["y"], 
        "x" : ncl_prp["x"],
        } 
    
    # Run 
    sfactor = get_sfactor(stk_level, 0, ndim=1, pixel_size=pixel_size)
    ncl_dst["asc_all_dst"] = _get_ncl_distances(
        (asc_msk > 0) & (asc_msk < 3), sfactor)
    ncl_dst["asc_spk_dst"] = _get_ncl_distances(
        asc_msk == 1, sfactor)
    ncl_dst["asc_nsp_dst"] = _get_ncl_distances(
        asc_msk == 2, sfactor)
    
    return ncl_dst

def bin_ncl_distances(
    ncl_dst, ncl_cls, ncl_classes,
    bin_size=4, bin_max=32, normalize=False, 
    stk_level=1, pixel_size=0.5,
    ):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_counts(ncl_cls, ncl_classes, normalized=False):
        counts = np.bincount(
            ncl_cls, minlength=len(ncl_classes)).astype("float32")
        total = counts.sum()
        if normalized and total > 0:
            return counts / total
        return counts
    
    def concatenate(dst_bin) :
        dst_bin = pd.concat(dst_bin, axis=1)
        dst_bin = dst_bin.loc[:, ~dst_bin.columns.duplicated(keep="first")]
        return dst_bin
    
    # Execute -----------------------------------------------------------------
    
    ncl_dst_bin = []
    ncl_cls = ncl_cls["class"]
    for tag in ["all", "spk", "nsp"]:
        
        # Rescale
        dst = ncl_dst[f"asc_{tag}_dst"]
        
        # Define bins
        bins = np.arange(bin_size, bin_max + bin_size, bin_size)
        
        # Get binned distances
        dst_bin = []
        for b, d in enumerate(bins):
            d0 = 0 if b == 0 else bins[b - 1]
            dst_bin_cls = ncl_cls[(dst >= d0) & (dst < d)]
            dst_bin.append(get_counts(
                dst_bin_cls, ncl_classes, normalized=True))
        
        # Normalization (enrichment)
        if normalize:
            dst_bin_all = np.stack(get_counts(
                ncl_cls, ncl_classes, normalized=True))
            for b, d in enumerate(bins):
                dst_bin[b] = dst_bin[b] / dst_bin_all
                    
        # Convert to dataframe
        dst_bin = pd.DataFrame(dst_bin)
        dst_bin.columns = [clss + f"_asc_{tag}_dst" for clss in ncl_classes]
        dst_bin.insert(0, "bins", bins)
        
        # Append
        ncl_dst_bin.append(dst_bin)
        
    # Convert to df
    ncl_dst_bin = concatenate(ncl_dst_bin)
            
    return ncl_dst_bin

def average_ncl_distances_bin(ncl_dst_bin_mrg):
    ncl_dst_bin_mrg = pd.concat(ncl_dst_bin_mrg)
    ncl_dst_bin_avg = ncl_dst_bin_mrg.groupby(level=0, sort=False).mean()
    ncl_dst_bin_std = ncl_dst_bin_mrg.groupby(level=0, sort=False).std()
    ncl_dst_bin_sem = ncl_dst_bin_mrg.groupby(level=0, sort=False).sem()   
    return ncl_dst_bin_avg, ncl_dst_bin_std, ncl_dst_bin_sem

#%% function(s) : asc stats ---------------------------------------------------
    
def get_asc_stats(
    pvw_msk, ncl_prp, asc_prp, asc_cls, vws_path,
    pvw_level=4, pixel_size=0.5,
    ): 

    # Init.
    asc_sts = {}
    spk_idx = np.where(asc_cls["class"] == 0)[0]
    nsp_idx = np.where(asc_cls["class"] == 1)[0]
    all_idx = np.union1d(spk_idx, nsp_idx)
    sfactor = get_sfactor(pvw_level, 0, ndim=2, pixel_size=pixel_size)
        
    # Areas
    tissue_area = np.sum(pvw_msk > 0) * sfactor
    asc_all_area = np.sum(asc_prp["area"][all_idx])
    asc_spk_area = np.sum(asc_prp["area"][spk_idx])
    asc_nsp_area = np.sum(asc_prp["area"][nsp_idx])
    asc_all_area_ratio = asc_all_area / tissue_area
    asc_spk_area_ratio = asc_spk_area / tissue_area  
    asc_nsp_area_ratio = asc_nsp_area / tissue_area  
    
    # Counts
    ncl_num = len(ncl_prp["y"])
    asc_all_num = len(all_idx)        
    asc_spk_num = len(spk_idx)
    asc_nsp_num = len(nsp_idx)
    asc_all_num_ratio = asc_all_num / ncl_num
    asc_spk_num_ratio = asc_spk_num / ncl_num
    asc_nsp_num_ratio = asc_nsp_num / ncl_num
  
    # Append
    asc_sts["tissue_area"       ] = tissue_area
    asc_sts["asc_all_area"      ] = asc_all_area
    asc_sts["asc_spk_area"      ] = asc_spk_area
    asc_sts["asc_nsp_area"      ] = asc_nsp_area
    asc_sts["asc_all_area_ratio"] = asc_all_area_ratio
    asc_sts["asc_spk_area_ratio"] = asc_spk_area_ratio
    asc_sts["asc_nsp_area_ratio"] = asc_nsp_area_ratio
    
    asc_sts["ncl_num"           ] = ncl_num
    asc_sts["asc_all_num"       ] = asc_all_num
    asc_sts["asc_spk_num"       ] = asc_spk_num
    asc_sts["asc_nsp_num"       ] = asc_nsp_num
    asc_sts["asc_all_num_ratio" ] = asc_all_num_ratio
    asc_sts["asc_spk_num_ratio" ] = asc_spk_num_ratio
    asc_sts["asc_nsp_num_ratio" ] = asc_nsp_num_ratio
    
    # Format
    asc_sts = pd.DataFrame([asc_sts])
    asc_sts.insert(0, "view", vws_path.name)
    
    return asc_sts

def average_asc_stats(asc_sts_mrg):
    asc_sts_avg = asc_sts_mrg.mean(numeric_only=True).to_frame().T
    asc_sts_std = asc_sts_mrg.mean(numeric_only=True).to_frame().T
    asc_sts_sem = asc_sts_mrg.mean(numeric_only=True).to_frame().T
    return asc_sts_avg, asc_sts_std, asc_sts_sem

#%% function(s) : format/avg. data --------------------------------------------

def format_ncl_data(ncl_prp, ncl_cls, ncl_int, ncl_dst, vws_path):
    
    n = len(ncl_prp["y"])
    ncl_int.pop("y") ; ncl_int.pop("x")
    ncl_dst.pop("y") ; ncl_dst.pop("x")
    ncl_data = {
        "view"         : [vws_path.name] * n,
        "y"            : ncl_prp["y"],
        "x"            : ncl_prp["x"],
        "area"         : ncl_prp["area"] ,
        "eccentricity" : ncl_prp["eccentricity"],
        "class"        : ncl_cls["class"],
        "entropy"      : ncl_cls["entropy"],
        }
    ncl_data.update(
        {k: v for k, v in ncl_int.items() if k not in ("x", "y")})
    ncl_data.update(
        {k: v for k, v in ncl_dst.items() if k not in ("x", "y")})
    
    return pd.DataFrame(ncl_data)

def format_asc_data(asc_prp, asc_cls, vws_path):
    
    spk_idx = np.where(asc_cls["class"] == 0)[0]
    nsp_idx = np.where(asc_cls["class"] == 1)[0]
    all_idx = np.union1d(spk_idx, nsp_idx)
    
    n = len(all_idx)
    asc_data = {
        "view"         : [vws_path.name] * n,
        "y"            : asc_prp["y"][all_idx],
        "x"            : asc_prp["x"][all_idx],
        "area"         : asc_prp["area"][all_idx],
        "eccentricity" : asc_prp["eccentricity"][all_idx],
        "class"        : asc_cls["class"][all_idx],
        "entropy"      : asc_cls["entropy"][all_idx],
        }
    
    return pd.DataFrame(asc_data)

def average_data(data, classes, vws_path):  

    # Nested function(s) ------------------------------------------------------        

    def _average_data(grp, method="mean"):
        
        # Statistics
        if method == "mean": df = grp.mean(numeric_only=True)
        if method == "std" : df = grp.std(numeric_only=True)
        if method == "sem" : df = grp.sem(numeric_only=True)
            
        # Count & frequency
        if "y" in data.columns:
            count = grp.size()
            freq = count / len(data)
        else:
            count = grp["count"].sum()
            freq = df["freq"]

        # Add count & frequency
        df = df.assign(count=count)
        df = df.assign(freq=freq)

        # Format
        if "y" in data.columns:
            df.index = df.index.map(lambda i: classes[i])
            df = df.reset_index(names="class").assign(view=vws_path.name)
        else:
            df = df.reset_index(names="class")

        cols = [c for c in ["view", "class", "count", "freq"] if c in df.columns]
        features = [c for c in df.columns if c not in cols]
        
        return df[cols + features]

    # Execute -----------------------------------------------------------------

    # Group per class
    if classes[0] == "Speckle":
        classes = classes.copy()
        classes.remove("Non-valid")
    dat = data.drop(columns=["x", "y"], errors="ignore")
    grp = dat.groupby("class", sort=True)
        
    # Format
    avg = _average_data(grp, method="mean")
    std = _average_data(grp, method="std")
    sem = _average_data(grp, method="sem")
    
    # Add "all" group (for asc)
    if classes[0] == "Speckle" and "y" in data.columns:
        classes = ["All"]
        dat["class"] = 0
        grp = dat.groupby("class", sort=True)
        avg_ = _average_data(grp, method="mean")
        std_ = _average_data(grp, method="std")
        sem_ = _average_data(grp, method="sem")
        avg = pd.concat([avg_, avg], ignore_index=True)
        std = pd.concat([std_, std], ignore_index=True)
        sem = pd.concat([sem_, sem], ignore_index=True)
        
    return avg, std, sem

#%% function(s) : get_plot() --------------------------------------------------

def get_plot(
    ncl_all, ncl_avg, ncl_sem,
    asc_all, asc_avg, asc_sem, 
    dst_avg, dst_sem, asc_sts,
    ncl_classes, ncl_colors, int_norm,
    ):
    
    # Inputs ------------------------------------------------------------------

    schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
    sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
    merged = ncl_all["view"].nunique(dropna=False) > 1

    hline_params = {
        "y" : 1, "color" : "0.75", "zorder" : 0,
        "linewidth" : 1, "linestyle" : ":",
        }
    
    grid_params = {
        "axis" : "x", "color" : "0.75", "zorder" : 0,
        "linewidth" : 1, "linestyle" : ":", 
        }

    # Nested function(s) ------------------------------------------------------
    
    def _ylim(ax, min_ymax=6):
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, np.maximum(min_ymax, ymax))
    
    def _barplot(
            ax, avg, x, err=None,
            width=0.5, shift=0.0, color="gray"
            ):
        
        x_pos = np.arange(len(x)) + shift
        ax.bar(
            x_pos, avg, width=width,
            color=color, capsize=5, zorder=2,
            )
        
        if not err is None:
            ax.errorbar(
                x_pos, avg, yerr=np.vstack([err, err]),
                capsize=3, elinewidth=0.5, ecolor="0.5", fmt="none",
                )
            
        ax.axhline(**hline_params)
        ax.grid(**grid_params)
    
    def _boxplot(ax, vals, x, merged=False, color="gray"):
        
        ax.boxplot(
            vals, positions=[x], widths=0.5, patch_artist=True,
            boxprops=dict(color="0.5", facecolor="none", linewidth=1),
            medianprops=dict(color="0.5", linewidth=1),
            whiskerprops=dict(color="0.5", linewidth=1),
            capprops=dict(color="0.5", linewidth=1),
            showfliers=False,
            )
        
        # Add dots
        x_jittered = x + np.random.normal(0, 0.08, size=len(vals))
        
        ax.scatter(
            x_jittered, vals,
            color=color, alpha=0.75, s=10, edgecolors="none",
            )
        
        ax.axhline(**hline_params)
        ax.grid(**grid_params)    
    
    def _lineplot(ax, vals, x, color, label):
        ax.plot(x, vals, color=color, label=label)
        ax.grid(**grid_params)
    
    # Execute -----------------------------------------------------------------
    
    # plt.ioff()
    
    # Initialize plot
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 12))
    axes = axes.flatten()
    
    # Main title/info ---------------------------------------------------------
       
    # Main title    
    fig.suptitle(
        "Merged views" if merged else ncl_all["view"].iloc[0],
        fontsize=16, fontweight="bold", color="k",
        y=0.92, x=0.025, ha="left",
        )

    # Secondary text
    text = (
        f"n views = {ncl_all['view'].nunique()} | "
        f"n nuclei = {ncl_all['count'].sum():,} | "
        f"avg. entropy = {ncl_all['entropy'].mean():.3f}"
        )
    fig.text(
        0.025, 0.89, text,
        fontsize=12, color="0.5",
        ha="left", va="top",
        )
    
    # (0) Class predictions frequency -----------------------------------------
    
    ax = axes[0]

    for cl, cla in enumerate(ncl_classes):
        _barplot(
            ax, ncl_avg["freq"], sclasses, err=ncl_sem["freq"], 
            color=ncl_colors, 
            )
        
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Ncl. class pred. freq.", pad=10)
    
    # (1) Class predictions entropy -------------------------------------------
    
    ax = axes[1]
    
    for cl, cla in enumerate(ncl_classes):
        vals = np.array(ncl_all[ncl_all["class"] == cla]["entropy"])
        _boxplot(ax, vals, cl, color=ncl_colors[cl])
        
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Entropy")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Ncl. class pred. entropy", pad=10)
    
    # (2) ASC statistics ------------------------------------------------------ 
    
    ax = axes[2]
    
    tags = ["nsp_area", "spk_area", "nsp_num", "spk_num"]
    for t, tag in enumerate(tags):
        _boxplot(ax, asc_sts[f"asc_{tag}_ratio"], t, color="gray")
        if t in [1, 3]:
            if t == 1: label = "area ratio"
            if t == 3: label = "numb ratio"
            ax.text(
                t - 0.5, 0.095, label,
                fontsize=10, color="black",
                horizontalalignment="center",
                verticalalignment="top",
                )
        
    # Legend
    ax.set_ylim(0, 0.10)
    ax.set_ylabel("ratio")
    ax.set_xticks(np.arange(0, len(tags)))
    ax.set_xticklabels(
        ["non\nspk", "spk", "non\nspk", "spk"], 
        rotation=45, ha="center",
        )
    ax.set_title("ASC statistics", pad=10)
        
    # (3) Class ASC intensities -----------------------------------------------
    
    ax = axes[3]
        
    for t, tag in enumerate(["nsp", "spk"]):
        shift = -0.2 if t == 0 else 0.2
        for cl, cla in enumerate(ncl_classes):
            _barplot(
                ax, ncl_avg[f"asc_{tag}_int"], sclasses, 
                err=ncl_sem[f"asc_{tag}_int"],
                width=0.3, shift=shift, color=ncl_colors, 
                )
            
    ax.text(
        0.5, 0.95, "non-speckle vs. speckle",
        fontsize=10, color="black",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        )
            
    # Legend
    _ylim(ax)
    ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    if int_norm:
        norm_str = "\n norm. int."
    ax.set_title(f"Ncl. ASC int.{norm_str}", pad=10)
    
    # (4) ASC distance class enrichment ---------------------------------------
    
    ax = axes[4]
    
    for cl, cla in enumerate(ncl_classes):
        x = dst_avg["bins"]
        vals = dst_avg[cla + "_asc_nsp_dst"]
        _lineplot(ax, vals, x, ncl_colors[cl], cla)
        
    # Legend
    ax.set_ylabel("Fold change")
    ax.set_xlabel("ASC cluster dist. (pixels)")
    ax.set_xticks(x)
    if int_norm:
        norm_str = "\n norm. dist."
    ax.set_title(f"Non-speckle dist. to ncl.{norm_str}", pad=10)
    
    # (5) ASC distance class enrichment ---------------------------------------
    
    ax = axes[5]
    
    for cl, cla in enumerate(ncl_classes):
        x = dst_avg["bins"]
        vals = dst_avg[cla + "_asc_spk_dst"]
        _lineplot(ax, vals, x, ncl_colors[cl], cla)
        
    # Legend
    ax.set_ylabel("Fold change")
    ax.set_xlabel("ASC cluster dist. (pixels)")
    ax.set_xticks(x)
    if int_norm:
        norm_str = "\n norm. dist."
    ax.set_title(f"Speckle dist. to ncl.{norm_str}", pad=10)
    
    # (6 - 11) Channel intensities --------------------------------------------
    
    for cl, cla in enumerate(ncl_classes):

        ax = axes[cl + 6]                   
        
        for ch, chn in enumerate(schannels):
            vals = np.array(ncl_all[ncl_all["class"] == cla][f"{chn}_int"])
            _boxplot(ax, vals, ch, color=ncl_colors[cl])
                    
        # Legend
        _ylim(ax)
        ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
        ax.set_xticklabels(schannels, rotation=45, ha="center")
        if int_norm:
            norm_str = "\nnorm. int."
        ax.set_title(cla + norm_str, pad=10)
    
    # -------------------------------------------------------------------------
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.close(fig) 
    # plt.ion()
    
    pass


#%% function(s) : pad_images() ------------------------------------------------

def pad_images(imgs):
    imgs_padded = []
    max_y = np.max([img.shape[0] for img in imgs])
    max_x = np.max([img.shape[1] for img in imgs])
    for img in imgs:
        pad_y = max_y - img.shape[0]
        pad_x = max_x - img.shape[1]
        top, bottom = pad_y // 2, pad_y - (pad_y // 2)
        left, right = pad_x // 2, pad_x - (pad_x // 2)
        padding = ((top, bottom), (left, right))
        imgs_padded.append(
            np.pad(img, padding, mode="constant", constant_values=0))
    return imgs_padded