#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import get_mask_from_coords

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QWidget, QLabel,
    )

#%% Class : AnnotateNCL() -----------------------------------------------------

class AnnotateNCL:
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
        
#%% Class(AnnotateNCL) : initialize() -----------------------------------------

    def initialize(self):
        
        # Paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
        # Maps
        clrs = self.ncl_colors
        clss = self.ncl_classes
        clrs.insert(0, [0.2, 0.2, 0.2])
        clss.insert(0, "NaN")
        self.ncl_class_map = {
            "cls_nan" : {"tag" : 0, "color" : clrs[0], "class" : clss[0]},
            "cls_oth" : {"tag" : 1, "color" : clrs[1], "class" : clss[1]},
            "cls_epi" : {"tag" : 2, "color" : clrs[2], "class" : clss[2]},
            "cls_neu" : {"tag" : 3, "color" : clrs[3], "class" : clss[3]},
            "cls_mac" : {"tag" : 4, "color" : clrs[4], "class" : clss[4]},
            "cls_bcl" : {"tag" : 5, "color" : clrs[5], "class" : clss[5]},
            "cls_tcl" : {"tag" : 6, "color" : clrs[6], "class" : clss[6]},
            }
        self.color_map = {
            v["tag"]: v["color"] for v in self.ncl_class_map.values()}
                
#%% Class(AnnotateNCL) : function(s) ------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        stk_nrm_path = vws_path / "stack_norm.tif"
        ncl_prp_path = vws_path / "nuclei_properties.pkl"
        self.ncl_ann_path = vws_path / "nuclei_annotations.pkl"
        
        # Load data
        self.stk_nrm = io.imread(stk_nrm_path)
        with open(ncl_prp_path, "rb") as file:
            self.ncl_prp = pickle.load(file)
        self.ncl_msk = get_mask_from_coords(
            self.ncl_prp["coords"], shape=self.stk_nrm[0, ...].shape)
        self.ncl_ctd = np.column_stack((
            self.ncl_prp["y"], self.ncl_prp["x"]))
        self.ncl_slc = np.zeros(self.ncl_ctd.shape[0], dtype="uint8")
        
        # Load nuclei classes (if exists)
        if not self.ncl_ann_path.exists():
            self.ncl_ann = {
                "y"      : [],
                "x"      : [],
                "idxs"   : [],
                "coords" : [], 
                "class"  : [],
                }
        else:
            
            with open(self.ncl_ann_path, "rb") as file:
                self.ncl_ann = pickle.load(file)
                
            # Update asc_slc
            for i, idx in enumerate(self.ncl_ann["idxs"]):
                self.ncl_slc[idx] = self.ncl_ann["class"][i]
                
    def update_select(self, tag):
                
        # Fetch & update selected points
        idxs = list(self.vwr.layers["points"].selected_data)
        
        if not idxs:
            return   
        
        # Update points layer
        self.ncl_slc[idxs] = tag
        self.vwr.layers["points"].features["class_label"] = self.ncl_slc
        self.vwr.layers["points"].refresh_colors()
        self.vwr.layers["points"].selected_data = set()
                
        # Update and save ncl_ann
        idxs = np.where(self.ncl_slc != 0)[0]
        self.ncl_ann["y"] = self.ncl_ctd[idxs, 0]
        self.ncl_ann["x"] = self.ncl_ctd[idxs, 1]
        self.ncl_ann["idxs"] = idxs
        self.ncl_ann["coords"] = [self.ncl_prp["coords"][idx] for idx in idxs]
        self.ncl_ann["class"] = self.ncl_slc[idxs]
        with open(self.ncl_ann_path, "wb") as file:
            pickle.dump(self.ncl_ann, file)
        self.get_info()
                
    def auto_contrast(self):
        img = self.vwr.layers["stk_nrm"].data 
        v0 = np.percentile(img, 0.01)
        v1 = np.percentile(img, 99.99)
        self.vwr.layers["stk_nrm"].contrast_limits = [v0, v1]

    def update_view(self):
        self.load_data()
        self.vwr.layers["stk_nrm"].data = self.stk_nrm
        self.vwr.layers["points" ].data = self.ncl_ctd
        self.vwr.layers["points" ].features = {"class_label": self.ncl_slc}
        self.update_select(0)
        self.auto_contrast()
        self.get_info()
        
    def update_channel(self):
        self.chn_idx = self.vwr.dims.current_step[0]
        self.get_info()
           
    # Buttons & shortcuts -----------------------------------------------------
    
    def prev_view(self):
        if self.vws_idx > 0:
            self.vws_idx -= 1
        self.update_view()
        
    def next_view(self):
        if self.vws_idx < (len(self.vws_paths) - 1):
            self.vws_idx += 1
        self.update_view()
        
    def hide_mask(self):
        self.vwr.layers["points"].visible = 0
        
    def show_mask(self):
        self.vwr.layers["points"].visible = 1
        
#%% Class(AnnotateNCL) : init_viewer() ----------------------------------------       

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
        self.cls_group_box = QGroupBox("AnnotateNCL")
        self.cls_group_layout = QVBoxLayout()
        for key, val in self.ncl_class_map.items():
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
        for key, val in self.ncl_class_map.items():
            button = getattr(self, key)
            button.clicked.connect(
                lambda checked, i=val["tag"] : self.update_select(i))

        # "update_channel()"
        self.vwr.dims.events.current_step.connect(self.update_channel)
        
        # Shortcuts -----------------------------------------------------------
        
        @self.vwr.bind_key("PageDown", overwrite=True)
        def _prev_view(vwr):
            self.prev_view()
                    
        @self.vwr.bind_key("PageUp", overwrite=True)
        def _next_view(vwr):
            self.next_view()
        
        @self.vwr.bind_key("End", overwrite=True)
        def _hide_labels_key(viewer):
            self.hide_mask()
            yield
            self.show_mask()
                        
#%% Class(AnnotateNCL) : init_layers() ----------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.stk_nrm, name="stk_nrm", visible=1,
            opacity=0.6, gamma=0.6, 
            )
        self.auto_contrast()
                
        self.vwr.add_points(
            self.ncl_ctd, name="points", visible=1,
            blending="translucent", size=5, border_width=0.0,
            features={"class_label": self.ncl_slc},
            face_color="class_label",
            face_color_cycle=self.color_map,
            opacity=1.0, 
            )
        self.vwr.layers["points"].mode = "select"
        self.vwr.layers["points"].highlight_box = False # debug
                        
#%% Class(AnnotateNCL) : info() -----------------------------------------------

    def get_info(self):
        
        channel = self.channels[self.chn_idx]
        
        classify_str = []
        for key, val in self.ncl_class_map.items():    
            classify_str.append(
                f"{val['class']} : {np.sum(self.ncl_ann['class'] == val['tag'])}\n"
                )
        
        self.info.setText(
            f"{self.vws_paths[self.vws_idx].name}\n"
            f"({channel})\n"
            f"{''.join(classify_str)}"
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    ann = AnnotateNCL(main)
    ncl_prp = ann.ncl_prp
    ncl_ann = ann.ncl_ann