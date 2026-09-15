#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import load_compressed_stack

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QWidget, QLabel,
    )

#%% Class : Classify() --------------------------------------------------------

class Classify:
    def __init__(self, main):
        self.main = main
        self.parameters = main.parameters
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
        
#%% Class(Classify) : initialize() --------------------------------------------

    def initialize(self):
        
        # Paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
        # Maps
        self.classes.insert(0, "NaN")
        self.colors.insert(0, [0.2, 0.2, 0.2])
        self.class_map = {
            "cls_nan" : {"tag" : 0, "color" : self.colors[0], "class" : self.classes[0]},
            "cls_oth" : {"tag" : 1, "color" : self.colors[1], "class" : self.classes[1]},
            "cls_epi" : {"tag" : 2, "color" : self.colors[2], "class" : self.classes[2]},
            "cls_neu" : {"tag" : 3, "color" : self.colors[3], "class" : self.classes[3]},
            "cls_mac" : {"tag" : 4, "color" : self.colors[4], "class" : self.classes[4]},
            "cls_bcl" : {"tag" : 5, "color" : self.colors[5], "class" : self.classes[5]},
            "cls_tcl" : {"tag" : 6, "color" : self.colors[6], "class" : self.classes[6]},
            }
        self.color_map = {
            v["tag"]: v["color"] for v in self.class_map.values()}
                
#%% Class(Classify) : function(s) ----------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
        ncl_ann_path = vws_path / "nuclei_classes_annotations.pkl"
        stk_raw_path = vws_path / "stack_norm.tif"
        
        # Load data
        with open(ncl_ctd_path, "rb") as file:
            self.ncl_ctd = pickle.load(file)
            self.ncl_ctd = np.column_stack((
                self.ncl_ctd["y"][self.stk_level],
                self.ncl_ctd["x"][self.stk_level],
                ))
        self.stk_raw = io.imread(stk_raw_path)
        
        # Load nuclei classes (if exists)
        if not ncl_ann_path.exists():
            self.ncl_ann = np.full(self.ncl_ctd.shape[0], 0)
        else:
            with open(ncl_ann_path, "rb") as file:
                self.ncl_ann = pickle.load(file)
                        
    def auto_contrast(self):
        img = self.vwr.layers["image"].data 
        v0 = np.percentile(img, 0.01)
        v1 = np.percentile(img, 99.99)
        self.vwr.layers["image"].contrast_limits = [v0, v1]

    def update_view(self):
        self.load_data()
        self.vwr.layers["image" ].data = self.stk_raw
        self.vwr.layers["points"].data = self.ncl_ctd
        self.update_classes(0)
        self.auto_contrast()
        
    def update_channel(self):
        self.chn_idx = self.vwr.dims.current_step[0]
        self.get_info()
           
    # Buttons -----------------------------------------------------------------
    
    def prev_view(self):
        if self.vws_idx > 0:
            self.vws_idx -= 1
        self.update_view()
        
    def next_view(self):
        if self.vws_idx < (len(self.vws_paths) - 1):
            self.vws_idx += 1
        self.update_view()
        
    def update_classes(self, tag):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        ncl_ann_path = vws_path / "nuclei_classes_annotations.pkl"
        
        # Fetch & update selected points
        idxs = list(self.vwr.layers["points"].selected_data)
        self.ncl_ann[idxs] = tag
        self.vwr.layers["points"].features["class_label"] = self.ncl_ann
        self.vwr.layers["points"].refresh_colors()
        self.vwr.layers["points"].selected_data = set()
        self.get_info()
        
        if not idxs:
            return    
        
        # Save
        with open(ncl_ann_path, "wb") as file:
            pickle.dump(self.ncl_ann, file)
        
#%% Class(Classify) : init_viewer() -------------------------------------------            

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
        
        # Create "classify" menu
        self.cls_group_box = QGroupBox("Classify")
        self.cls_group_layout = QVBoxLayout()
        for key, val in self.class_map.items():
            setattr(self, key, QPushButton(val["class"]))
            self.cls_group_layout.addWidget(getattr(self, key))
        self.cls_group_box.setLayout(self.cls_group_layout)
        
        # Create texts
        self.info = QLabel()
        self.info.setFont(QFont("Consolas"))

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vws_group_box)
        self.layout.addWidget(self.cls_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Callsbacks ----------------------------------------------------------
    
        # "view"
        self.vws_prv.clicked.connect(self.prev_view)
        self.vws_nxt.clicked.connect(self.next_view)
        
        # "classify"
        for key, val in self.class_map.items():
            button = getattr(self, key)
            button.clicked.connect(
                lambda checked, i=val["tag"] : self.update_classes(i))

        # "update_channel()"
        self.vwr.dims.events.current_step.connect(self.update_channel)
                        
#%% Class(Classify) : init_layers() -------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.stk_raw, name="image", visible=1,
            opacity=0.6, gamma=0.6, 
            )
        self.auto_contrast()
                
        self.vwr.add_points(
            self.ncl_ctd, name="points", visible=1,
            blending="translucent", size=5, border_width=0.0,
            features={"class_label": self.ncl_ann},
            face_color="class_label",
            face_color_cycle=self.color_map,
            opacity=1.0, 
            )
        self.vwr.layers["points"].mode = "select"
                        
#%% Class(Classify) : info() --------------------------------------------------

    def get_info(self):
        
        channel = self.channels[self.chn_idx]
        
        classify_str = []
        for key, val in self.class_map.items():    
            classify_str.append(
                f"{val['class']} : {np.sum(self.ncl_ann == val['tag'])}\n"
                )
        
        self.info.setText(
            f"{self.vws_paths[self.vws_idx].name}\n"
            f"({channel})\n"
            f"{''.join(classify_str)}"
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    classify = Classify(main)