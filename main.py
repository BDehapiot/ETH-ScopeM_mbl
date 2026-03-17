#%% Imports -------------------------------------------------------------------

import shutil
from slide import Slide

#%% Class(Extract) ------------------------------------------------------------

class Extract:
    def __init__(self, data_path):
        self.data_path = data_path
        
        # Run
        self.initialize()
        self.extract()
        
#%% Class(Extract) initialize() -----------------------------------------------

    def initialize(self):
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))

#%% Class(Extract) extract() --------------------------------------------------

    def extract(self):
        
        # Setup output directory
        self.out_paths = []
        for path in self.slide_paths:
            out_path = self.data_path / path.stem
            if out_path.exists():
                shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            self.out_paths.append(out_path)
            
        # Save preview (level = 5)
        for p, path in enumerate(self.slide_paths):
            print(path.name)
            slide = Slide(path)
            metadata = slide.metadata
            for view in range(len(metadata["positions"])):
                if len(metadata["levels"]) > 4:
                    slide.load(channels="all", level=5, view=view)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports 
    import time
    from pathlib import Path
    
    # -------------------------------------------------------------------------
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    
    # -------------------------------------------------------------------------
    
    extract = Extract(data_path)
    # paths = extract.paths