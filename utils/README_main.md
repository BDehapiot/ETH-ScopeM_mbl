## Overview

The `main.py` script and its associated `function.py` file contain the core 
code for performing automated multi-channel immunofluorescence slide analysis 
on the `.qptiff` files located in the specified data folder. The procedure is 
divided into sequential stages that must be run in order to complete the 
analysis. Individual sample results are saved in dedicated subfolders, while 
the averaged results are saved in the root data folder. 

Additionally, `display.py` & `plot.py` are used to visualize the final results 
via Napari and Matplotlib, respectively, while the `annotate_[...].py` & 
`train_[...].py` scripts handle training the deep-learning models used for 
object segmentation and classification in the raw images.

Images must be aquired at the same magnification (pixel size).  
Images must contain the following channels in the correct order :

``` yml
- chn 1 : DAPI  
- chn 2 : Ly6G  
- chn 3 : TCRbeta  
- chn 4 : CD19  
- chn 5 : ASC  
- chn 6 : Iba1  
- chn 7 : EpCAM  
- chn 8 : Autofluo. (AF)  
```

Nuclei classification is performed across the following categories :
``` yml
- Other       : # Unclassified / negative for lineage markers
- Epithelium  : # High EpCAM expression
- Neutrophil  : # High Ly6G expression (EpCAM-low)
- Macrophage  : # High Iba1 expression
- Bcell       : # High CD19 expression
- Tcell       : # High TCRbeta expression
```

ASC cluster classification is performed across the following categories : 
``` yml
- Speckle     : # Small, bright, and isolated ASC cluster
- Non-speckle : # Large, bright ASC cluster
- Non-valid   : # Artifactual or noisy detection (excluded)
```

## Parameters

### Procedure
Control code execution.  
0 to skip, 1 to run if not already done and 2 to force execution.
``` yml
"force_extract"         : int (0, 1 or 2)
"force_predict_nuclei"  : int (0, 1 or 2)
"force_predict_asc"     : int (0, 1 or 2)
"force_classify_nuclei" : int (0, 1 or 2)
"force_classify_asc"    : int (0, 1 or 2)
"force_measure"         : int (0, 1 or 2)
"force_analyse"         : int (0, 1 or 2)
```

### Paths
Provide path toward the data directory. 
``` yml
"data_path"    : str (e.g. "D:\local_mbl\data")
```

### Segmentation
Control Nuclei & ASC clusters segmentation from predicted probability images.
``` yml
# Nuclei
"ncl_thresh_0" : float (0 to 1) # probability thresh. 0 (outlines)
"ncl_thresh_1" : float (0 to 1) or None # probability thresh. 1 (cores)
"ncl_min_size" : int or None # min. area (in µm² at level = 0)
"ncl_max_size" : int or None # max. area (in µm² at level = 0)
    
# ASC
"asc_thresh_0" : float (0 to 1)  # probability thresh. 0 (outlines)
"asc_thresh_1" : float (0 to 1) or None # probability thresh. 1 (cores)
"asc_min_size" : int or None # min. area (in µm² at level = 0)
"asc_max_size" : int or None # max. area (in µm² at level = 0)
```
2 thresholds approach (object separation) works for edt models only.  
Use `None` for thresh_1 with binary models.

### Measurments

#### Nuclei fluorescence intensities
Fluorescence channel intensities are measured in a square region of size 
`int_crop_size` around the nucleus centroid. If normalized (`int_norm = True`), 
measured values dividing them by the average value across all nuclei in the 
same sample.
``` yml
"pixel_size"    : float # image resolution (in µm at level = 0)
"int_crop_size" : int   # fluo. intensities window size (in µm at level = 0)
"int_norm"      : bool  # normalize fluo. intensities (per channel)
```

#### Nuclei distances to ASC
Nuclei distances to the closest ASC cluster (non-speckle or speckle) are 
measured and binned according to `dst_bin_size` and `dst_bin_max`. The results
show the proportion of a nuclei class whose closest ASC cluster falls within 
the given distance bin. If normalized (`dst_norm = True`), the measured values 
are divided by the overall baseline class frequencies across all bins to compute spatial enrichment.
``` yml
"dst_bin_size"  : int   # distance bin size (in µm at level = 0)
"dst_bin_max"   : int   # distance bin max (in µm at level = 0)
"dst_norm"      : bool  # normalize distances (per class)
```

### Analysis
Add name of the view(s) that will be skipped during the merged analysis.
``` yml
"skip_views" : [
    "251127_D2_S2_Scan1_unmixed_01",
    ], 
```

## Outputs

### For each view

#### `preview_[raw/norm/mask].tif`
Low resolution raw, normalized and masked view of the sample. 

#### `stack_norm.tif`
High resolution normalized view of the sample.

#### `nuclei_data.csv`
Individual nuclei measurments.
``` yml
"view"         : str   # view name
"Y"            : int   # y position (in pixels at level = 1)
"X"            : int   # x position (in pixels at level = 1)
"area"         : int   # area (in µm² at level = 0)
"eccentricity" : float # shape eccentricity (0 to 1, 0 being perfect circle)
"class"        : int   # predicted class (as integer)
"entropy"      : float # class prediction entropy

# Fluo. int. measurments (from a square region centered on nuclei)
"DAPI_int"     : float # (A.U.) from DAPI channel 
"Ly6G_int"     : float # (A.U.) from Ly6G channel
"TCRbeta_int"  : float # (A.U.) from TCRbeta channel
"CD19_int"     : float # (A.U.) from CD19 channel
"Iba1_int"     : float # (A.U.) from Iba1 channel
"EpCAM_int"    : float # (A.U.) from EpCAM channel
"AF_int"       : float # (A.U.) from AF channel
"asc_all_int"  : float # (A.U.) from ASC channel (all)
"asc_spk_int"  : float # (A.U.) from ASC channel (speckle)
"asc_nsp_int"  : float # (A.U.) from ASC channel (non-speckle)

# Distance from the closest ASC cluster
"asc_all_dst"  : float # all ASC cluster 
"asc_spk_dst"  : float # speckle ASC cluster
"asc_nsp_dst"  : float # non-speckle ASC cluster
```
Fluorescence intensity measurements for the ASC channel are performed on 
processed images, with all pixels outside the predicted ASC mask set to 0.

#### `nuclei_data_[avg/std/sem].csv`
Avg/std/sem nuclei measurments.  
Measurments are similar to `nuclei_data.csv` except :
``` yml
"class" : str   # predicted class (as string)
"count" : int   # nuclei count for a given class
"freq"  : float # nuclei freq for a given class
```

#### `nuclei_distances_bin.csv`
Binned nuclei distance to closest ASC cluster.
``` yml
"bins"  : int  # distance bin (in µm at level = 0)

# Binned nuclei distance to the closest all ASC cluster
"Other_asc_all_dst"      : float # "Other" class nuclei
"Epithelium_asc_all_dst" : float # "Epithelium" class nuclei
"Neutrophil_asc_all_dst" : float # "Neutrophil" class nuclei
"Macrophage_asc_all_dst" : float # "Macrophage" class nuclei
"Bcell_asc_all_dst"      : float # "Bcell"class nuclei
"Tcell_asc_all_dst"      : float # "Tcell" class nuclei

# Binned nuclei distance to the closest speckle ASC cluster
"Other_asc_spk_dst"      : float # "Other" class nuclei
"Epithelium_asc_spk_dst" : float # "Epithelium" class nuclei
"Neutrophil_asc_spk_dst" : float # "Neutrophil" class nuclei
"Macrophage_asc_spk_dst" : float # "Macrophage" class nuclei
"Bcell_asc_spk_dst"      : float # "Bcell"class nuclei
"Tcell_asc_spk_dst"      : float # "Tcell" class nuclei

# Binned nuclei distance to the closest non-speckle ASC cluster
"Other_asc_nsp_dst"      : float # "Other" class nuclei
"Epithelium_asc_nsp_dst" : float # "Epithelium" class nuclei
"Neutrophil_asc_nsp_dst" : float # "Neutrophil" class nuclei
"Macrophage_asc_nsp_dst" : float # "Macrophage" class nuclei
"Bcell_asc_nsp_dst"      : float # "Bcell"class nuclei
"Tcell_asc_nsp_dst"      : float # "Tcell" class nuclei
```

#### `asc_data.csv`
Individual ASC cluster measurments.
``` yml
"view"         : str   # view name
"Y"            : int   # y position (in pixels at level = 1)
"X"            : int   # x position (in pixels at level = 1)
"area"         : int   # area (in µm² at level = 0)
"eccentricity" : float # shape eccentricity (0 to 1, 0 being perfect circle)
"class"        : int   # predicted class (as integer)
"entropy"      : float # class prediction entropy
```

#### `asc_data_[avg/std/sem].csv`
Avg/std/sem ASC cluster measurments.  
Measurments are similar to `asc_data.csv` except :
``` yml
"class" : str   # predicted class (as string)
"count" : int   # ASC cluster count for a given class
"freq"  : float # ASC cluster freq for a given class
```

#### `asc_stats.csv`
``` yml
"view"               : str   # view name
"tissue_area"        : int   # tissue area (in µm² at level = 0)
"asc_all_area"       : int   # cum. ASC area (in µm² at level = 0)
"asc_spk_area"       : int   # cum. speckle area (in µm² at level = 0)
"asc_nsp_area"       : int   # cum. non-speckle area (in µm² at level = 0)
"asc_all_area_ratio" : float # tissue / cum. ASC area ratio
"asc_spk_area_ratio" : float # tissue / cum. speckle area ratio
"asc_nsp_area_ratio" : float # tissue / cum. non-speckle area ratio
"ncl_num"            : int   # nuclei count
"asc_all_num"        : int   # ASC count
"asc_spk_num"        : int   # speckle count
"asc_nsp_num"        : int   # non-speckle count
"asc_all_num_ratio"  : float # nuclei / ASC count ratio
"asc_spk_num_ratio"  : float # nuclei / speckle count ratio
"asc_nsp_num_ratio"  : float # nuclei / non-speckle count ratio
 ```

### For the all dataset

#### `nuclei_data_avg_[merged/avg/std/sem].csv`
Merged, avg, std, sem of individual view `nuclei_data.csv`.  

#### `nuclei_distances_bin_[avg/std/sem].csv`
Avg, std, sem of individual view `nuclei_distances_bin.csv`.  

#### `asc_data_avg_[merged/avg/std/sem].csv`
Merged, avg, std, sem of individual view `asc_data_avg.csv`. 

#### `asc_sts_[merged/avg/std/sem].csv`
Merged, avg, std, sem of individual view `asc_sts.csv`. 

Dataset statistics are made on averaged view data, not individual nuclei 
or ASC clusters.

