#%% Imports -------------------------------------------------------------------

from pathlib import Path
from functions import load_metadata, load_arr

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
        
    }

#%% Class(DataSet) ------------------------------------------------------------

class DataSet:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.initialize()
        
#%% Class(DataSet) : initialize() ---------------------------------------------

    def initialize(self):
        self.slide_paths = list(self.data_path.glob("*unmixed.qptiff"))
        for path in self.slide_paths:
            metadata = load_metadata(path)
            for view in range(len(metadata["positions"])):
                print(view)
        # self.slide_stems = [path.stem[:12] for path in self.slide_paths]

#%% Class(DataSet) : load() ---------------------------------------------------

    # def load(self, path, channels="all", level=5, view=0):
    #     metadata = load_metadata()
    #     arr = load_arr(
    #         self.path, metadata=self.metadata, 
    #         channels=channels, level=level, view=view
    #         ) 
    #     return 

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    dataset = DataSet(parameters=parameters)
    