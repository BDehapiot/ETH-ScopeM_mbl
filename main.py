#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from qptifffile import QPTiffFile

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_mbl\data")
# file_name = "251126_D7_S2_Scan1.unmixed.qptiff"
# file_name = "251126_E2_S2_Scan1.unmixed.qptiff"
# file_name = "251126_E6_S2_Scan1.unmixed.qptiff"
file_name = "251126_E7_S2_Scan1.unmixed.qptiff"

#%% Function(s) ---------------------------------------------------------------

def get_views(file_path):
    
    global preview
    
    # Load preview
    preview = []
    with QPTiffFile(file_path) as qptiff:
        for channel in qptiff.get_biomarkers():
            preview.append(qptiff.read_region(channel, level=4))
    preview = np.mean(np.stack(preview), axis=0)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    file_path = data_path / file_name
    
    # get_views()
    get_views(file_path)

    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(preview)