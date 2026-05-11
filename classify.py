#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import load_compressed_stack

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton,
    QWidget, QLabel,
    )

#%% Inputs --------------------------------------------------------------------

classify_parameters = {
        
    # Paths
    "data_path" : Path("D:\local_mbl\data"),

    }

#%% Class : Classify() --------------------------------------------------------

class Classify:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Initialize
        self.vws_idx = 0
        self.chn_idx = 0      

        # Run
        self.initialize()
        self.init_viewer()
        self.init_layers()
        
#%% Class(Classify) : initialize() ---------------------------------------------

    def initialize(self):
        
        # Paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
#%% Class(Classify) : function(s) ----------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        mtd_path = self.mtd_paths[self.vws_idx]
        vws_path = self.vws_paths[self.vws_idx]
        pvw_its_path = vws_path / "preview_intensities.pkl"
        ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
        stk_raw_path = vws_path / "stack_raw.npz"
        
        # Load
        with open(mtd_path, "rb") as file:
            self.mtd = pickle.load(file)
        with open(pvw_its_path, "rb") as file:
            self.pvw_its = pickle.load(file)
        with open(ncl_ctd_path, "rb") as file:
            self.ncl_ctd = pickle.load(file)
            self.ncl_ctd = np.column_stack((
                self.ncl_ctd["y"][self.stk_level],
                self.ncl_ctd["x"][self.stk_level],
                ))
        self.stk_raw = load_compressed_stack(stk_raw_path)
                        
    def auto_contrast(self):
        img = self.vwr.layers["image"].data 
        v0 = np.percentile(img, 0.01)
        v1 = np.percentile(img, 99.99)
        self.vwr.layers["image"].contrast_limits = [v0, v1]
                
    def change_channel(self):
        self.chn_idx = self.vwr.dims.current_step[0]
        self.get_info()
    
    def update_view(self):
        self.load_data()
        self.vwr.layers["image" ].data = self.stk_raw[self.chn_idx]
        self.vwr.layers["points"].data = self.ncl_ctd
        self.auto_contrast()
                
    # Buttons -----------------------------------------------------------------
    
    def prev_view(self):
        if self.vws_idx > 0:
            self.vws_idx -= 1
        self.update_view()
    
    def next_view(self):
        if self.vws_idx < (len(self.vws_paths) - 1):
            self.vws_idx += 1
        self.update_view()
        
#%% Class(Display) : init_viewer() --------------------------------------------                

    def init_viewer(self):
            
        # Viewer --------------------------------------------------------------
        
        self.vwr = napari.Viewer()
        
        # Create "image" menu
        self.vws_group_box = QGroupBox("Select view")
        self.vws_group_layout = QHBoxLayout()
        self.vws_prv = QPushButton("prev")
        self.vws_nxt = QPushButton("next")
        self.vws_group_layout.addWidget(self.vws_prv)
        self.vws_group_layout.addWidget(self.vws_nxt)
        self.vws_group_box.setLayout(self.vws_group_layout)
        self.vws_prv.clicked.connect(self.prev_view)
        self.vws_nxt.clicked.connect(self.next_view)

        # Create texts
        self.info = QLabel()
        self.info.setFont(QFont("Consolas"))

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vws_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Callsbacks ----------------------------------------------------------
    
        self.vwr.dims.events.current_step.connect(self.change_channel)
        
        # Shortcuts -----------------------------------------------------------
            
        
#%% Class(Classify) : init_layers() -------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.stk_raw, name="image", visible=1,
            gamma=0.75, 
            )
                
        self.vwr.add_points(
            self.ncl_ctd, name="points", visible=1,
            blending="additive", face_color="orange",
            size=5, border_width=0.0,
            opacity=0.25, 
            )
                
        self.auto_contrast()
        
#%% Class(Classify) : info() --------------------------------------------------

    def get_info(self):
        
        channel = self.channels[self.chn_idx]
        self.info.setText(
            f"channel : {channel}"
            )
        

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    classify = Classify(main, parameters=classify_parameters)
    
#%%

    stk_raw = classify.stk_raw