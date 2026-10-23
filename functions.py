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
from skimage import draw
from skimage.transform import rescale
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, binary_closing, remove_small_objects

# scipy
from scipy.stats import pearsonr
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
        prd, thresh_0=0.25, thresh_1=0.50, min_size=8, 
        prd_level=0, msk_level=1,
        ):
    
    # Initialize
    rf = 1 / (2 ** (msk_level - prd_level))
    min_size = min_size // ((2 ** msk_level) ** 2) 
    
    # Rescale prediction (if required)
    if rf != 1: 
        prd = rescale(prd, rf)
        
    # Get mask
    if thresh_1 is not None:
        msk_0 = prd > thresh_0
        msk_1 = prd > thresh_1
        msk = watershed(-prd, label(msk_1), mask=msk_0)
        msk[lbl_conn(msk, conn=2) == 2] = 0
        msk = msk > 0
    else:
        msk = prd > thresh_0
    
    # Remove small objects
    if min_size > 0:
        msk = remove_small_objects(msk, min_size=min_size)
    
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

def get_obj_properties(nrm, msk):

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
        "y"      : list(props["centroid-0"].astype("uint16")),  
        "x"      : list(props["centroid-1"].astype("uint16")),  
        "area"   : list(props["area"].astype("uint16")), 
        "coords" : [coord.astype("uint16") for coord in props["coords"]], 
        "eccentricity"   : list(props["eccentricity"].astype("float32")), 
        "intensity_max"  : list(props["intensity_max"].astype("float32")),  
        "intensity_min"  : list(props["intensity_min"].astype("float32")), 
        "intensity_mean" : list(props["intensity_mean"].astype("float32")), 
        "intensity_std"  : list(props["intensity_std"].astype("float32")), 
        }  

#%% function(s) : get_obj_crops() ---------------------------------------------

def get_obj_crops(stk, ys, xs, crop_size=32, crop_fraction=None):

    # Check inputs
    ndim = stk.ndim
    if isinstance(ys, list): ys = np.array(ys)
    if isinstance(xs, list): xs = np.array(xs)
    
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
    
# def get_obj_crops(stk, ctd, crop_size=32, crop_fraction=None, stk_level=1):
    
#     # Pad (acc. to crop half size)
#     chs = crop_size // 2 # crop half size
#     pad_width = ((0, 0), (chs, chs), (chs, chs))
#     stk = np.pad(stk, pad_width=pad_width, mode="reflect")

#     # Extract coordinates
#     ys = ctd["y"][stk_level].astype(int)
#     xs = ctd["x"][stk_level].astype(int)
    
#     # Select random coordinates
#     if crop_fraction is not None:
#         crop_idx = np.random.choice(
#             np.arange(0, len(ys)), 
#             size=int(len(ys) * crop_fraction), 
#             replace=False,
#             )
#         ys = ys[crop_idx]
#         xs = xs[crop_idx]
    
#     # Extract crops
#     crops = []
#     for (y, x) in zip(ys, xs):
#         y0 = y; y1 = y + crop_size
#         x0 = x; x1 = x + crop_size
#         if y1 - y0 != crop_size: print(y0, y1)
#         if x1 - x0 != crop_size: print(x0, x1)
#         crops.append(stk[..., y0:y1, x0:x1])
        
#     return np.stack(crops)

def get_crops_processed(crp):
    crp_prc = np.delete(crp, (0, 4, -1), axis=1) # remove DAPI, ASC & autofluo channels
    crp_prc = np.transpose(crp_prc, (0, 2, 3, 1)) # transposition
    crp_prc = crp_prc.astype("float32") / 255 # normalize
    return crp_prc
    
#%% function(s) : get_obj_intensities() ---------------------------------------

def get_obj_intensities(crp, channels, normalize=True):
    obj_int = {}
    for c, chn in enumerate(channels):
        _obj_int = np.mean(crp[..., c], axis=(1, 2))
        if normalize:
            _obj_int /= np.mean(_obj_int) 
        obj_int[f"{chn}_int"] = list(_obj_int)
    return obj_int

#%% function(s) : get_asc_distances() -----------------------------------------

def get_asc_distances(asc_msk, ncl_prp, stk_level=1):
            
    # Get edt
    asc_edt = distance_transform_edt(~asc_msk)
    
    # Get asc distances
    asc_dst = asc_edt[(
        np.array(ncl_prp["y"]).astype(int), 
        np.array(ncl_prp["x"]).astype(int),
        )]
    
    # Scale acc. to stk_level
    asc_dst *= 2 ** stk_level
    
    return asc_dst

#%% function(s) : format data -------------------------------------------------

def format_ncl_data(ncl_prp, ncl_cls, ncl_int, asc_dst, vws_path):
    
    n = len(ncl_prp["y"])
    ncl_int.pop("y") ; ncl_int.pop("x")
    asc_dst.pop("y") ; asc_dst.pop("x")
    ncl_data = {
        "view"         : [vws_path.name] * n,
        "y"            : ncl_prp["y"],
        "x"            : ncl_prp["x"],
        "area"         : ncl_prp["area"],
        "eccentricity" : ncl_prp["eccentricity"],
        "class"        : ncl_cls["class"],
        "entropy"      : ncl_cls["entropy"],
        }
    ncl_data.update(
        {k: v for k, v in asc_dst.items() if k not in ("x", "y")})
    ncl_data.update(
        {k: v for k, v in ncl_int.items() if k not in ("x", "y")})
    
    return pd.DataFrame(ncl_data)

def format_asc_data(asc_prp, asc_cls, asc_int, vws_path):
    
    n = len(asc_prp["y"])
    asc_int.pop("y") ; asc_int.pop("x")
    asc_data = {
        "view"         : [vws_path.name] * n,
        "y"            : asc_prp["y"],
        "x"            : asc_prp["x"],
        "area"         : asc_prp["area"],
        "eccentricity" : asc_prp["eccentricity"],
        "class"        : asc_cls["class"],
        "entropy"      : asc_cls["entropy"],
        }
    asc_data.update(
        {k: v for k, v in asc_int.items() if k not in ("x", "y")})
    
    return pd.DataFrame(asc_data)

#%% function(s) : get_class_avg() ---------------------------------------------

def get_class_avg(data, classes, vws_path):  

    def _get_class_avg(grp, method="mean"):
        
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

    # Group per class
    grp = data.drop(columns=["x", "y"], errors="ignore")
    grp = grp.groupby("class", sort=True)
        
    # Format
    avg = _get_class_avg(grp, method="mean")
    std = _get_class_avg(grp, method="std")
    sem = _get_class_avg(grp, method="sem")

    return avg, std, sem

#%% function(s) : plot --------------------------------------------------------





#%% function(s) : nuclei ------------------------------------------------------
    
# def get_nuclei_stats(ncl_crp, channels, vws_path):        
#     ncl_sts = {"avg" : {}}
#     for c, chn in enumerate(channels):
#         ncl_sts["avg"][chn] = np.mean(ncl_crp[:, c, ...], axis=(1, 2))
#     return ncl_sts

def get_nuclei_intensities(ncl_crp, channels):        
    ncl_int = {}
    for c, chn in enumerate(channels):
        ncl_int[f"{chn}_int"] = list(np.mean(ncl_crp[:, c, ...], axis=(1, 2)))
    return ncl_int

def get_nuclei_stats(ncl_crp, ncl_cls, vws_path, channels, classes):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_nuclei_intensities(ncl_crp, channels, vws_path):        
        ncl_int = {}
        for c, chn in enumerate(channels):
            ncl_int[f"{chn}_int"] = np.mean(ncl_crp[:, c, ...], axis=(1, 2))
        return ncl_int
    
    # def get_mean_norm(df):
    #     dfn = df.copy()
    #     idx = df.filter(regex='|'.join(["entropy", "avg"])).columns
    #     dfn[idx] = df[idx] / df[idx].mean()
    #     idx = df.filter(regex='|'.join(["count"])).columns
    #     dfn[idx] = df[idx] / df[idx].sum()
    #     return dfn
    
    # def get_class_avg(df):
    #     dfa = df.copy()
    #     dfa = df.groupby("class").mean(numeric_only=True)
    #     dfa = dfa.drop(columns=["idx"])
    #     dfa.insert(loc=0, column="count", value=df.groupby("class").size())
    #     dfa = dfa.reset_index()
    #     dfa.insert(loc=0, column="view", value=[vws_path.name] * n_classes)
    #     return dfa
    
    # Execute -----------------------------------------------------------------
    
    # 
    ncl_sts = {"view" : [vws_path.name] * ncl_crp.shape[0]}
    
    # Get nuclei intensities
    ncl_sts.update(get_nuclei_intensities(ncl_crp, channels, vws_path))
    
    # Add view, class & entropy 
    ncl_sts["class"  ] = ncl_cls["class"  ]
    ncl_sts["entropy"] = ncl_cls["entropy"]
    
    ncl_sts = pd.DataFrame(ncl_sts)
    
    # # Extrat data
    # ncl_df = pd.DataFrame()
    # n_nuclei = len(ncl_cls["y"])
    # n_classes = len(classes)
    # ncl_df["view"   ] = [vws_path.name] * n_nuclei
    # ncl_df["idx"    ] = np.arange(n_nuclei)
    # ncl_df["class"  ] = np.array(classes)[ncl_prd]
    # ncl_df["entropy"] = ncl_ent
    # for msr in ["avg"]:
    #     for chn in channels:
    #         key = f"{chn}_{msr}"
    #         ncl_df[key] = ncl_sts[msr][chn]
    # ncl_dfa  = class_avg(ncl_df)
    # ncl_dfn  = mean_norm(ncl_df)
    # ncl_dfan = mean_norm(ncl_dfa)
        
    return ncl_sts

# def format_nuclei_stats(
#     ncl_prd, ncl_ent, ncl_sts, vws_path, 
#     channels, classes
#     ):
    
#     def mean_norm(df):
#         dfn = df.copy()
#         idx = df.filter(regex='|'.join(["entropy", "avg"])).columns
#         dfn[idx] = df[idx] / df[idx].mean()
#         idx = df.filter(regex='|'.join(["count"])).columns
#         dfn[idx] = df[idx] / df[idx].sum()
#         return dfn
    
#     def class_avg(df):
#         dfa = df.copy()
#         dfa = df.groupby("class").mean(numeric_only=True)
#         dfa = dfa.drop(columns=["idx"])
#         dfa.insert(loc=0, column="count", value=df.groupby("class").size())
#         dfa = dfa.reset_index()
#         dfa.insert(loc=0, column="view", value=[vws_path.name] * n_classes)
#         return dfa
    
#     # Extrat data
#     ncl_df = pd.DataFrame()
#     n_nuclei = ncl_prd.shape[0]
#     n_classes = len(classes)
#     ncl_df["view"   ] = [vws_path.name] * n_nuclei
#     ncl_df["idx"    ] = np.arange(n_nuclei)
#     ncl_df["class"  ] = np.array(classes)[ncl_prd]
#     ncl_df["entropy"] = ncl_ent
#     for msr in ["avg"]:
#         for chn in channels:
#             key = f"{chn}_{msr}"
#             ncl_df[key] = ncl_sts[msr][chn]
#     ncl_dfa  = class_avg(ncl_df)
#     ncl_dfn  = mean_norm(ncl_df)
#     ncl_dfan = mean_norm(ncl_dfa)
        
#     return ncl_df, ncl_dfa, ncl_dfn, ncl_dfan

#%% function(s) : asc ---------------------------------------------------------

# def get_asc_stats(
#         pvw_msk, ncl_ctd, asc_msk,
#         pvw_level=4, stk_level=1
#         ):
    
#     # Init.
#     asc_sts = {}
#     pvw_factor = (2 ** pvw_level) ** 2
#     asc_factor = (2 ** stk_level) ** 2
    
#     # Areas
#     all_area = np.sum(pvw_msk > 0) * pvw_factor
#     asc_area = np.sum(asc_msk > 0) * asc_factor
#     asc_area_nrm = asc_area / all_area
    
#     # Counts
#     ncl_num = len(ncl_ctd["y"][0])
#     asc_num = np.max(label(asc_msk > 0))
#     asc_spk_num = np.max(label(asc_msk > 200))
#     asc_num_ratio = asc_num / ncl_num
#     asc_spk_num_ratio = asc_spk_num / ncl_num
            
#     # Append
#     asc_sts["all_area"         ] = all_area
#     asc_sts["asc_area"         ] = asc_area
#     asc_sts["asc_area_ratio"   ] = asc_area_nrm
#     asc_sts["ncl_num"          ] = ncl_num
#     asc_sts["asc_num"          ] = asc_num
#     asc_sts["asc_spk_num"      ] = asc_spk_num
#     asc_sts["asc_num_ratio"    ] = asc_num_ratio
#     asc_sts["asc_spk_num_ratio"] = asc_spk_num_ratio
    
#     return asc_sts

def get_asc_stats(
    pvw_msk, ncl_prp, asc_prp, asc_cls,
    vws_path, pvw_level=4, stk_level=1, 
    ):
    
    # Init.
    asc_sts = {}
    pvw_factor = (2 ** pvw_level) ** 2
    asc_factor = (2 ** stk_level) ** 2
        
    # Areas
    all_area = np.sum(pvw_msk > 0) * pvw_factor
    asc_area = np.sum(asc_prp["area"]) * asc_factor
    asc_area_ratio = asc_area / all_area
    
    # Counts
    ncl_num = len(ncl_prp["y"])
    asc_num = len(asc_prp["y"])
    spk_num = (asc_cls["class"] == 0).sum()
    asc_num_ratio = asc_num / ncl_num
    spk_num_ratio = spk_num / ncl_num
            
    # Append
    asc_sts["all_area"       ] = all_area
    asc_sts["asc_area"       ] = asc_area
    asc_sts["asc_area_ratio" ] = asc_area_ratio
    asc_sts["ncl_num"        ] = ncl_num
    asc_sts["asc_num"        ] = asc_num
    asc_sts["spk_num"        ] = spk_num
    asc_sts["asc_num_ratio"  ] = asc_num_ratio
    asc_sts["spk_num_ratio"  ] = spk_num_ratio
    
    # Format
    asc_sts = pd.DataFrame([asc_sts])
    asc_sts.insert(0, "view", vws_path.name)
    
    return asc_sts

def format_asc_stats(asc_sts, vws_path):
    asc_sts_df = pd.DataFrame([asc_sts])
    asc_sts_df.insert(0, "view", vws_path.name)
    return asc_sts_df

# def get_asc_distances(asc_msk, ncl_prp, classes, stk_level=1):
            
#     # Get edt
#     asc_edt = distance_transform_edt(~asc_msk)
    
#     # Get asc distances
#     asc_dst = asc_edt[(
#         np.array(ncl_prp["y"]).astype(int), 
#         np.array(ncl_prp["x"]).astype(int),
#         )]
    
#     # Scale acc. to stk_level
#     asc_dst *= 2 ** stk_level
    
    return asc_dst

def format_asc_distances(
    asc_dst, ncl_prd, classes, bin_size=10, bin_max=200, stk_level=1):
    
    def get_counts(ncl_prd, normalized=False):
        counts = np.bincount(ncl_prd, minlength=len(classes)).astype("float32")
        total = counts.sum()
        if normalized and total > 0:
            return counts / total
        return counts
    
    # Get bins
    bins = np.arange(bin_size, bin_max + bin_size, bin_size)
    
    # Count nuclei class
    counts_raw = []
    counts_all_raw = np.stack(get_counts(ncl_prd, normalized=True))
    for d, dst in enumerate(bins):
        dst0 = 0 if d == 0 else bins[d - 1]
        ncl_prd_dst = ncl_prd[(asc_dst >= dst0) & (asc_dst < dst)]
        counts_raw.append(get_counts(ncl_prd_dst, normalized=True))
        
    # Normalizations (enrichment)
    counts_nrm = []
    for d, dst in enumerate(bins):
        counts_nrm.append(counts_raw[d] / counts_all_raw)
    
    # Get dataframe
    asc_dst_df = pd.DataFrame(counts_raw, columns=classes)
    asc_dst_df.insert(0, "distances", bins)
    asc_dst_dfn = pd.DataFrame(counts_nrm, columns=classes)
    asc_dst_dfn.insert(0, "distances", bins)
    
    return asc_dst_df, asc_dst_dfn

#%% function(s) : plots -------------------------------------------------------

def plot(
    ncl_df, ncl_dfa, ncl_dfn, ncl_dfan, asc_dst_df, asc_dst_dfn,
    asc_bin_size, asc_bin_max, 
    channels, classes, colors
    ):
    
    # Inputs ------------------------------------------------------------------
    
    schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
    sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
    merged = ncl_df["view"].nunique(dropna=False) > 1
    
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
        
    def _boxplot(ax, vals, x, color, merged=False):
        ax.boxplot(
            vals, positions=[x], widths=0.5, patch_artist=True,
            boxprops=dict(color="0.5", facecolor="none", linewidth=1),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color="0.5", linewidth=1),
            capprops=dict(color="0.5", linewidth=1),
            showfliers=False,
            )
        ax.axhline(**hline_params)
        ax.grid(**grid_params)
        
    def _barplot(ax, avg, x, color, err=None):
        ax.bar(x, avg, width=0.5, capsize=5, color=color, zorder=2)
        if not err is None:
            ax.errorbar(
                x, avg, yerr=np.vstack([err, err]),
                capsize=3, elinewidth=0.5, ecolor="0.5", fmt="none",
                )
        ax.axhline(**hline_params)
        ax.grid(**grid_params)

    def _lineplot(ax, vals, x, color, label):
        ax.plot(x, vals, color=color, label=label)
        ax.grid(**grid_params)
    
    # Execute -----------------------------------------------------------------
    
    plt.ioff()
    
    # Initialize plot
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 12))
    axes = axes.flatten()
    
    # Main title/info ---------------------------------------------------------
       
    # Main title    
    fig.suptitle(
        "Merged views" if merged else ncl_dfa["view"].iloc[0],
        fontsize=16, fontweight="bold", color="k",
        y=0.92, x=0.025, ha="left",
        )

    # Secondary text
    text = (
        f"n views = {ncl_dfa['view'].nunique()} | "
        f"n nuclei = {ncl_dfa['count'].sum():,} | "
        f"avg. entropy = {ncl_dfa['entropy'].mean():.3f}"
        )
    fig.text(
        0.025, 0.89, text,
        fontsize=12, color="0.5",
        ha="left", va="top",
        )
    
    # (0) Class predictions frequency -----------------------------------------
    
    ax = axes[0]
    dfs = ncl_dfan
    
    avg, std, sem = [], [], []
    for cl, cla in enumerate(classes):
        avg.append(dfs[dfs["class"] == cla]["count"].mean())
        std.append(dfs[dfs["class"] == cla]["count"].std())
        sem.append(dfs[dfs["class"] == cla]["count"].sem())
    if merged:
        _barplot(ax, avg, sclasses, colors, err=sem)
    else:
        _barplot(ax, avg, sclasses, colors, err=None)
    
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class frequency", pad=10)
    
    # (1) Class predictions entropy -------------------------------------------
    
    ax = axes[1]
    dfs = ncl_df
    
    for cl, cla in enumerate(classes):
        vals = np.array(dfs[dfs["class"] == cla]["entropy"])
        _boxplot(ax, vals, cl, colors[cl], merged=merged)
        
    # Legend
    ax.set_ylim(0, 1)
    ax.set_ylabel("Entropy")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class pred. entropy", pad=10)
    
    # (2) ??? -----------------------------------------------------------------
    
    ax = axes[2]
    ax.axis("off")
    ax.set_facecolor("none")
    
    # (3 - 8) Channel intensities ---------------------------------------------
    
    for cl, cla in enumerate(classes):

        ax = axes[cl + 3]                   
        dfs = ncl_dfan if merged else ncl_dfn
        
        for ch, chn in enumerate(schannels):
            vals = np.array(dfs[dfs["class"] == cla][f"{chn}_avg"])
            _boxplot(ax, vals, ch, colors[cl], merged=merged)
                    
        # Legend
        _ylim(ax)
        ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
        ax.set_xticklabels(schannels, rotation=45, ha="center")
        ax.set_title(cla, pad=10)
    
    # (9) Class ASC intensities -----------------------------------------------
    
    ax = axes[9]
    dfs = ncl_dfan if merged else ncl_dfn
    
    avg, std, sem = [], [], []
    for cl, cla in enumerate(classes):
        avg.append(dfs[dfs["class"] == cla]["ASC_avg"].mean())
        std.append(dfs[dfs["class"] == cla]["ASC_avg"].std())
        sem.append(dfs[dfs["class"] == cla]["ASC_avg"].sem())
    _barplot(ax, avg, sclasses, colors, err=sem)
        
    # Legend
    _ylim(ax)
    ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
    ax.set_xticks(np.arange(0, len(sclasses)))
    ax.set_xticklabels(sclasses, rotation=45, ha="center")
    ax.set_title("Class ASC int.", pad=10)
        
    # (10) ASC distance class frequency ---------------------------------------
    
    ax = axes[10]
    dfs = asc_dst_df
    
    for cl, cla in enumerate(classes):
        vals = dfs[cla]
        dsts = dfs["distances"]
        _lineplot(ax, vals, dsts, colors[cl], cla)
        
    # Legend
    ax.set_ylabel("Frequency")
    ax.set_xlabel("ASC cluster dist. (pixels)")
    ax.set_xticks(dsts)
    ax.set_title("ASC dist. class freq.", pad=10)
    
    # (11) ASC distance class enrichment --------------------------------------
       
    ax = axes[11]
    dfs = asc_dst_dfn
    
    for cl, cla in enumerate(classes):
        vals = dfs[cla]
        dsts = dfs["distances"]
        _lineplot(ax, vals, dsts, colors[cl], cla)
        
    # Legend
    ax.set_ylabel("Fold change")
    ax.set_xlabel("ASC cluster dist. (pixels)")
    ax.set_xticks(dsts)
    ax.set_title("ASC dist. class enrichment", pad=10)
    
    # -------------------------------------------------------------------------
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.close(fig) 
    plt.ion()

    return fig

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