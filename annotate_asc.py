#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from main import Main, main_parameters

# functions
from functions import imread_channel, get_mask_from_props

# skimage
from skimage.measure import label, regionprops_table

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton, QWidget, QLabel,
    )

#%% Class : AnnotateASC() -----------------------------------------------------

class AnnotateASC:
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
        
#%% Class(AnnotateASC) : initialize() -----------------------------------------

    def initialize(self):
        
        # Paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
#%% Class(AnnotateASC) : function(s) ------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        stk_nrm_path = vws_path / "stack_norm.tif"
        asc_prp_path = vws_path / "asc_properties.pkl"
        self.asc_cls_path = vws_path / "asc_classes.pkl"
        
        # Load data
        self.asc_nrm = imread_channel(stk_nrm_path, channel_idx=4)
        with open(asc_prp_path, "rb") as file:
            self.asc_prp = pickle.load(file)
        self.asc_msk = get_mask_from_props(
            self.asc_prp, shape=self.asc_nrm.shape)
        self.asc_slc = np.zeros_like(self.asc_msk, dtype="uint8")
        
        # Load asc classes (if exists)
        if not self.asc_cls_path.exists():
            self.asc_cls = {
                "y"      : [],
                "x"      : [],
                "coords" : [], 
                "class"  : [],
                }
        else:
            with open(self.asc_cls_path, "rb") as file:
                self.asc_cls = pickle.load(file)
                
            # Update asc_slc
            for coords, clss in zip(self.asc_cls["coords"], self.asc_cls["class"]):
                self.asc_slc[tuple(coords.T)] = clss
                
    def update_mask(self):
        
        # Update asc_slc
        asc_msk = self.vwr.layers["asc_msk"].data
        asc_slc = self.vwr.layers["asc_slc"].data
        asc_slc[asc_msk == 0] = 0 
        self.vwr.layers["asc_slc"].data = asc_slc
        
        # Update asc_cls
        props = regionprops_table(
            label(asc_slc > 0), 
            intensity_image=asc_slc, 
            properties=("label", "centroid", "coords", "intensity_max")
            )
        self.asc_cls["y"     ] = list(props["centroid-0"].astype("uint16")),  
        self.asc_cls["x"     ] = list(props["centroid-1"].astype("uint16")),  
        self.asc_cls["coords"] = list(props["coords"])
        self.asc_cls["class" ] = [int(c) for c in props["intensity_max"]]
        
        # Save asc_cls
        with open(self.asc_cls_path, "wb") as file:
            pickle.dump(self.asc_cls, file)

        self.get_info()
        
    def update_view(self):
        self.load_data()
        self.vwr.layers["asc_nrm"].data = self.asc_nrm
        self.vwr.layers["asc_msk"].data = self.asc_msk
        self.vwr.layers["asc_slc"].data = self.asc_slc

    # Buttons -----------------------------------------------------------------
    
    def prev_view(self):
        if self.vws_idx > 0:
            self.vws_idx -= 1
        self.update_view()
        self.get_info()
        
    def next_view(self):
        if self.vws_idx < (len(self.vws_paths) - 1):
            self.vws_idx += 1
        self.update_view()
        self.get_info()
        
    def add_spk(self):
        self.vwr.layers["asc_slc"].selected_label = 1
    
    def rmv_spk(self):
        self.vwr.layers["asc_slc"].selected_label = 2
        
    # Shorcuts ----------------------------------------------------------------
        
    def hide_mask(self):
        self.vwr.layers["asc_msk"].visible = 0
        self.vwr.layers["asc_slc"].visible = 0
        
    def show_mask(self):
        self.vwr.layers["asc_msk"].visible = 1
        self.vwr.layers["asc_slc"].visible = 1
        
#%% Class(AnnotateASC) : init_viewer() ----------------------------------------        

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
        self.cls_add = QRadioButton("valid")
        self.cls_rmv = QRadioButton("invalid")
        self.cls_add.setChecked(True)
        self.cls_group_layout.addWidget(self.cls_add)
        self.cls_group_layout.addWidget(self.cls_rmv)
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
    
        self.vws_prv.clicked.connect(self.prev_view)
        self.vws_nxt.clicked.connect(self.next_view)
        self.cls_add.toggled.connect(self.add_spk)
        self.cls_rmv.toggled.connect(self.rmv_spk)
        
#%% Class(AnnotateASC) : shortcuts -------------------------------------------- 

        @self.vwr.bind_key("PageDown", overwrite=True)
        def _prev_view(vwr):
            self.prev_view()
                    
        @self.vwr.bind_key("PageUp", overwrite=True)
        def _next_view(vwr):
            self.next_view()
            
        @self.vwr.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.vwr.layers["asc_slc"].mode = "pan_zoom"
            yield
            self.vwr.layers["asc_slc"].mode = "paint"

        @self.vwr.bind_key("Delete", overwrite=True)
        def _hide_labels_key(viewer):
            self.hide_mask()
            yield
            self.show_mask()
            
        @Labels.bind_key("Enter", overwrite=True)
        def _save_mask_key(viewer):
            self.update_mask() 
            
        @self.vwr.mouse_drag_callbacks.append
        def mouse_actions(vwr, event):
            if event.button == 2:
                self.vwr.layers["asc_slc"].mode = "erase"
                yield
                self.vwr.layers["asc_slc"].mode = "paint"
        
#%% Class(AnnotateASC) : init_layers() ----------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.asc_nrm, name="asc_nrm", colormap="gray", visible=1,
            blending="minimum", opacity=0.75, gamma=0.5, 
            )
        
        self.vwr.add_image(
            self.asc_msk, name="asc_msk", colormap="yellow", visible=1,
            blending="additive", opacity=0.25,
            )
        
        self.vwr.add_labels(
            self.asc_slc, name="asc_slc", visible=1,
            blending="translucent", opacity=0.75, 
            )
        
        self.vwr.layers["asc_slc"].selected_label = 1
        self.vwr.layers["asc_slc"].brush_size = 10
        self.vwr.layers["asc_slc"].mode = "paint"
        self.get_info()

#%% Class(Classify) : info() --------------------------------------------------

    def get_info(self):

        n_add = self.asc_cls["class"].count(1)
        n_rmv = self.asc_cls["class"].count(2)

        self.info.setText(
            f"{self.vws_paths[self.vws_idx].name}\n"
            f"n valid   = {n_add}\n"
            f"n invalid = {n_rmv}"
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    annASC = AnnotateASC(main)
    asc_prp = annASC.asc_prp
    asc_cls = annASC.asc_cls
