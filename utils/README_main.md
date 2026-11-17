## Content

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

⚠️ Images must be aquired at the same magnification (pixel size).  
⚠️ Images must contain the following channels in the correct order :

```
- chn #1 : DAPI  
- chn #2 : Ly6G  
- chn #3 : TCRbeta  
- chn #4 : CD19  
- chn #5 : ASC  
- chn #6 : Iba1  
- chn #7 : EpCAM  
- chn #8 : Autofluo. (AF)  
```


## Main procedure

### 1) `Main.extract()`
Loads image and metadata from `.qptiff` files, automatically detect individual samples and normalize channel intensities 
tissue preview prediction, 
split individual views and normalizes image channels. 

### 2) `Main.predict_nuclei()` | `Main.predict_asc()`
Nuclei & ASC prediction employs deep learning segmentation models 
(U-Net via segmentation-models) to locate and mask nuclei (DAPI) and ASC 
positive structures.  

### 3) `Main.classify_nuclei()` | `Main.classify_asc()`
Classifies individual objects into defined categories
 
(e.g., cell phenotypes or structure subtypes) using convolutional neural network (CNN) classifiers.   Measurement (measure): Calculates spatial distances, fluorescence intensities per channel, areas, shapes, and regional statistics.   Analysis (analyse): Aggregates data across views and regions, computes statistical metrics (mean, standard deviation, standard error of the mean), and generates summary plots.   Note: .pkl files generated during execution are retained for internal development and state caching, and are therefore excluded from end-user reporting.