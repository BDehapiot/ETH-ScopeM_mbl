#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton,
    QWidget, QLabel,
    )

#%% Inputs --------------------------------------------------------------------

display_parameters = {
    
    # Channels
    "channels"  : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    
    }

#%% Class(Display) ------------------------------------------------------------

class Display:
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
                
#%% Class(Display) : initialize() ---------------------------------------------

    def initialize(self):
        
        # Paths
        self.sld_paths = self.main.sld_paths
        self.vws_paths = self.main.vws_paths
        self.stk_paths = [
            vws_path / (f"stack_level-{self.process_level}.tif")
            for vws_path in self.vws_paths
            ]
        self.msk_paths = [
            vws_path / (f"mask_level-{self.process_level}.tif")
            for vws_path in self.vws_paths
            ]
        self.prp_paths = [
            vws_path / (f"props_level-{self.process_level}.pkl")
            for vws_path in self.vws_paths
            ]
        
        # Variables
        self.chn = self.channels[self.chn_idx]

#%% Class(Display) : function(s) ----------------------------------------------

    # Logic & updates ---------------------------------------------------------
        
    def load_data(self):
        self.stk = io.imread(self.stk_paths[self.vws_idx])
        self.msk = io.imread(self.msk_paths[self.vws_idx])
        with open(self.prp_paths[self.vws_idx], "rb") as file:
            self.prp = pickle.load(file)
            self.pts = np.column_stack((self.prp["y"], self.prp["x"]))
        
    def auto_contrast(self):
        img = self.vwr.layers["image"].data 
        v0 = np.percentile(img,  0.01)
        v1 = np.percentile(img, 99.99)
        self.vwr.layers["image"].contrast_limits = [v0, v1]
        
    def update_view(self):
        self.load_data()
        self.vwr.layers["image" ].data = self.stk[self.chn_idx]
        self.vwr.layers["mask"  ].data = self.msk
        self.vwr.layers["points"].data = self.pts
        self.auto_contrast()
    
    def update_channel(self):    
        self.chn = self.chn_group.checkedButton().text()
        self.chn_idx = self.channels.index(self.chn)
        self.vwr.layers["image"].data = self.stk[self.chn_idx]
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
        
        # Create "view" menu
        self.vws_group_box = QGroupBox("Select view")
        self.vws_group_layout = QHBoxLayout()
        self.vws_prv = QPushButton("prev")
        self.vws_nxt = QPushButton("next")
        self.vws_group_layout.addWidget(self.vws_prv)
        self.vws_group_layout.addWidget(self.vws_nxt)
        self.vws_group_box.setLayout(self.vws_group_layout)
        self.vws_prv.clicked.connect(self.prev_view)
        self.vws_nxt.clicked.connect(self.next_view)
        
        # Create "channel" menu
        self.chn_group = QButtonGroup()
        self.chn_group_box = QGroupBox("Select channel")
        self.chn_group_layout = QVBoxLayout()
        for chn in self.channels:
            setattr(self, f"rad_{chn}", QRadioButton(f"{chn}"))
            if chn == self.channels[self.chn_idx]:
                getattr(self, f"rad_{chn}").setChecked(True)
            self.chn_group.addButton(getattr(self, f"rad_{chn}"))
            self.chn_group_layout.addWidget(getattr(self, f"rad_{chn}"))
            getattr(self, f"rad_{chn}").toggled.connect(self.update_channel)
        self.chn_group_box.setLayout(self.chn_group_layout)
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vws_group_box)
        self.layout.addWidget(self.chn_group_box)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts -----------------------------------------------------------

#%% Class(Display) : init_layers() --------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.stk[self.chn_idx], name="image", visible=1,
            gamma=0.75, 
            )
        self.vwr.add_image(
            self.msk, name="mask", visible=0,
            blending="additive", colormap="bop orange",
            opacity=0.5, 
            )
        
        self.vwr.add_points(
            self.pts, name="points", visible=1,
            blending="additive", face_color="orange",
            size=5, border_width=0.0,
            opacity=0.25, 
            )
                
        self.auto_contrast()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    display = Display(main, parameters=display_parameters)