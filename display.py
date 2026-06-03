#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import load_compressed_stack

# Qt
from qtpy.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton, QCheckBox, QWidget,
    )

#%% Inputs --------------------------------------------------------------------

display_parameters = {
        
    # Paths
    "data_path" : Path("D:\local_mbl\data"),

    # Maps
        
    "colors" : [
        [0.5, 0.5, 0.5], # Other
        [0.0, 1.0, 0.0], # Epithelium
        [1.0, 0.0, 0.0], # Neutrophil
        [1.0, 0.0, 1.0], # Macrophage
        [0.0, 1.0, 1.0], # Bcell
        [1.0, 1.0, 0.0], # Tcell
        ],

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
        self.cls_idx = 1

        # Run
        self.initialize()
        self.init_viewer()
        self.init_layers()
                
#%% Class(Display) : initialize() ---------------------------------------------

    def initialize(self):
        
        # Paths
        self.qpt_paths = self.main.qpt_paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
        # Maps
        
        self.class_map = {
            "cls_oth" : {"tag" : 0, "color" : self.colors[0], "class" : self.classes[0]},
            "cls_epi" : {"tag" : 1, "color" : self.colors[1], "class" : self.classes[1]},
            "cls_neu" : {"tag" : 2, "color" : self.colors[2], "class" : self.classes[2]},
            "cls_mac" : {"tag" : 3, "color" : self.colors[3], "class" : self.classes[3]},
            "cls_bcl" : {"tag" : 4, "color" : self.colors[4], "class" : self.classes[4]},
            "cls_tcl" : {"tag" : 5, "color" : self.colors[5], "class" : self.classes[5]},
            }
        
        self.color_map = {
            v["tag"]: v["color"] for v in self.class_map.values()}
                
#%% Class(Display) : function(s) ----------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        mtd_path = self.mtd_paths[self.vws_idx]
        vws_path = self.vws_paths[self.vws_idx]
        pvw_its_path = vws_path / "preview_intensities.pkl"
        ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
        ncl_cls_path = vws_path / "nuclei_classes_predict.pkl"
        stk_nrm_path = vws_path / "stack_norm.npz"
        
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
        with open(ncl_cls_path, "rb") as file:
            self.ncl_cls = pickle.load(file)
        self.ncl_cls = np.argmax(self.ncl_cls, axis=1)
        self.stk_nrm = load_compressed_stack(stk_nrm_path)

    def auto_contrast(self):
        img = self.vwr.layers["stk_nrm"].data 
        v0 = np.percentile(img, 0.01)
        v1 = np.percentile(img, 99.99)
        self.vwr.layers["stk_nrm"].contrast_limits = [v0, v1]
                
    def update_view(self):
        self.load_data()
        self.vwr.layers["stk_nrm"].data = self.stk_nrm[self.chn_idx]
        self.vwr.layers["ncl_ctd"].data = self.ncl_ctd
        self.vwr.layers["ncl_ctd"].features = {"class_label": self.ncl_cls}
        self.update_class()
        self.auto_contrast()
    
    def update_channel(self):    
        self.chn = self.chn_group.checkedButton().text()
        self.chn_idx = self.channels.index(self.chn)
        self.vwr.layers["stk_nrm"].data = self.stk_nrm[self.chn_idx]
        self.auto_contrast()
        
    def update_class(self):
        self.cll = [btn.text() for btn in self.cls_group.buttons() if btn.isChecked()]
        self.cls_idx = [self.classes.index(c) for c in self.cll]
        self.ncl_msk = np.isin(self.ncl_cls, self.cls_idx)
        self.vwr.layers["ncl_ctd"].shown = self.ncl_msk

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
        
        # Create "channel" menu
        self.chn_group = QButtonGroup()
        self.chn_group_box = QGroupBox("Select channel")
        self.chn_group_layout = QVBoxLayout()
        for c in self.channels:
            setattr(self, f"rad_{c}", QRadioButton(f"{c}"))
            if c == self.channels[self.chn_idx]:
                getattr(self, f"rad_{c}").setChecked(True)
            self.chn_group.addButton(getattr(self, f"rad_{c}"))
            self.chn_group_layout.addWidget(getattr(self, f"rad_{c}"))
            getattr(self, f"rad_{c}").toggled.connect(self.update_channel)
        self.chn_group_box.setLayout(self.chn_group_layout)
               
        # Create "class" menu
        self.cls_group = QButtonGroup()
        self.cls_group.setExclusive(False)
        self.cls_group_box = QGroupBox("Select class")
        self.cls_group_layout = QVBoxLayout()
        for c in self.classes:
            setattr(self, f"chk_{c}", QCheckBox(f"{c}")) 
            if c == self.classes[self.cls_idx]:
                getattr(self, f"chk_{c}").setChecked(True)
            self.cls_group.addButton(getattr(self, f"chk_{c}"))
            self.cls_group_layout.addWidget(getattr(self, f"chk_{c}"))
            getattr(self, f"chk_{c}").stateChanged.connect(self.update_class)
        self.cls_group_box.setLayout(self.cls_group_layout)
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vws_group_box)
        self.layout.addWidget(self.chn_group_box)
        self.layout.addWidget(self.cls_group_box)

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
            self.stk_nrm[self.chn_idx], name="stk_nrm", visible=1,
            opacity=0.6, gamma=0.6, 
            )
                
        self.vwr.add_points(
            self.ncl_ctd, name="ncl_ctd", visible=1,
            blending="translucent", size=8, border_width=0.0,
            features={"class_label": self.ncl_cls},
            face_color="class_label",
            face_color_cycle=self.color_map,
            opacity=1.0, 
            )
        
        self.update_class()
        self.auto_contrast()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    display = Display(main, parameters=display_parameters)