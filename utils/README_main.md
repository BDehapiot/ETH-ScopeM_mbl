## Overview

The pipeline is designed for automated multi-channel immunofluorescence slide 
analysis, running through sequential steps.

### `Main.extract()`
Loads raw `.qptiff` whole-slide images, performs tissue preview prediction, 
split individual views and normalizes image channels. 

### `Main.predict_nuclei()` | `Main.predict_asc()`
Nuclei & ASC prediction employs deep learning segmentation models (U-Net via segmentation-models) to locate and mask nuclei (DAPI) and ASC positive 
structures.  

### `Main.classify_nuclei()` | `Main.classify_asc()`
 Classifies individual objects into defined categories
 
  (e.g., cell phenotypes or structure subtypes) using convolutional neural network (CNN) classifiers.   Measurement (measure): Calculates spatial distances, fluorescence intensities per channel, areas, shapes, and regional statistics.   Analysis (analyse): Aggregates data across views and regions, computes statistical metrics (mean, standard deviation, standard error of the mean), and generates summary plots.   Note: .pkl files generated during execution are retained for internal development and state caching, and are therefore excluded from end-user reporting.