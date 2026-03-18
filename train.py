#%% Imports -------------------------------------------------------------------

from pathlib import Path

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    "patch_size"    : 512,
    "patch_overlap" : 256,
    
    }

#%% Class(Train) --------------------------------------------------------------

class Train:
    def __init__(self, data_path, parameters=None):
        self.data_path = data_path
        
        # Run
        self.initialize()
        
#%% Class(Train) : initialize() -----------------------------------------------

    def initialize(self):
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.slide_stems = [path.stem for path in self.slide_paths]

    
#%% Class(Train) : extract_patches() ------------------------------------------

    def extract_patches(self):
        self.train_path = Path.cwd() / "data" / "train"

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path("D:\local_mbl\data")
    train = Train(data_path, parameters=parameters)
    