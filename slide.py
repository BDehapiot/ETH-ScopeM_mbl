#%% Imports -------------------------------------------------------------------

from functions import load_metadata, load_arr

#%% Class(Slide) ---------------------------------------------------------------

class Slide:
    def __init__(self, path):
        self.path = path
        
        # Run
        self.initialize()
        
#%% Class(Slide) initialize() --------------------------------------------------

    def initialize(self):
        self.metadata = load_metadata(self.path)
        for key, val in self.metadata.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
#%% Class(Slide) load() --------------------------------------------------------

    def load(self, channels="all", level=5, view=0):
        return load_arr(
            self.path, metadata=self.metadata, 
            channels=channels, level=level, view=view
            )
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time
    from pathlib import Path
    
    # -------------------------------------------------------------------------
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    file_name = "251126_D7_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E2_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E6_S2_Scan1_unmixed.qptiff"
    # file_name = "251126_E7_S2_Scan1_unmixed.qptiff"
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
    channels = "DAPI"
    level = 5
    view = 1
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("Slide() : ", end="", flush=True)
    slide = Slide(path)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    metadata = slide.metadata
    
    t0 = time.time()
    print("Slide.load() : ", end="", flush=True)
    arr = slide.load(channels=channels, level=level, view=view)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(arr)
    