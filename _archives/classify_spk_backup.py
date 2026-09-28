#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# skimage
from skimage.segmentation import flood
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

#%% Class : AnnotateSPK() -----------------------------------------------------

class AnnotateSPK:
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
        
#%% Class(AnnotateSPK) : initialize() -----------------------------------------

    def initialize(self):
        
        # Paths
        self.mtd_paths = self.main.mtd_paths
        self.vws_paths = self.main.vws_paths
        
#%% Class(AnnotateSPK) : function(s) ------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        stk_nrm_path = vws_path / "stack_norm.tif"
        asc_msk_path = vws_path / "asc_mask.tif"
        self.spk_cls_path = vws_path / "speckle_classes.pkl"
        
        # Load data
        self.asc_nrm = io.imread(stk_nrm_path)[4, ...]
        self.asc_msk = io.imread(asc_msk_path)
        self.spk_msk = (self.asc_msk == 200).astype("uint16")
        self.spk_slc = np.zeros_like(self.asc_msk)
        
        # Load speckle classes (if exists)
        if not self.spk_cls_path.exists():
            self.spk_cls = {"c" : [], "y" : [], "x" : []}
        else:
            with open(self.spk_cls_path, "rb") as file:
                self.spk_cls = pickle.load(file)
                
            # Update spk_slc
            for c, y, x in zip(*self.spk_cls.values()):
                msk = flood(self.spk_msk, (y, x))
                self.spk_slc[msk] = c
                
    def update_mask(self):
        
        # Update spk_slc
        spk_msk = self.vwr.layers["spk_msk"].data
        spk_slc = self.vwr.layers["spk_slc"].data
        spk_slc[spk_msk == 0] = 0 
        self.vwr.layers["spk_slc"].data = spk_slc
        
        # Update spk_cls
        props = regionprops_table(
            label(spk_slc > 0), 
            intensity_image=spk_slc, 
            properties=("label", "centroid", "intensity_max")
            )
        self.spk_cls["c"] = list(props["intensity_max"])
        # self.spk_cls["y"] = list(props["centroid-0"])
        # self.spk_cls["x"] = list(props["centroid-1"])
        self.spk_cls["y"] = [round(y) for y in props["centroid-0"]]
        self.spk_cls["x"] = [round(x) for x in props["centroid-1"]]
        
        # Save spk_cls
        with open(self.spk_cls_path, "wb") as file:
            pickle.dump(self.spk_cls, file)

        self.get_info()
        
    def update_view(self):
        self.load_data()
        self.vwr.layers["asc_nrm"].data = self.asc_nrm
        self.vwr.layers["spk_msk"].data = self.spk_msk
        self.vwr.layers["spk_slc"].data = self.spk_slc

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
        self.vwr.layers["spk_slc"].selected_label = 1
    
    def rmv_spk(self):
        self.vwr.layers["spk_slc"].selected_label = 2
        
    # Shorcuts ----------------------------------------------------------------
        
    def hide_mask(self):
        self.vwr.layers["spk_msk"].visible = 0
        self.vwr.layers["spk_slc"].visible = 0
        
    def show_mask(self):
        self.vwr.layers["spk_msk"].visible = 1
        self.vwr.layers["spk_slc"].visible = 1
        
#%% Class(AnnotateSPK) : init_viewer() ----------------------------------------        

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
        
#%% Class(AnnotateSPK) : shortcuts -------------------------------------------- 

        @self.vwr.bind_key("PageDown", overwrite=True)
        def _prev_view(vwr):
            self.prev_view()
                    
        @self.vwr.bind_key("PageUp", overwrite=True)
        def _next_view(vwr):
            self.next_view()
            
        @self.vwr.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.vwr.layers["spk_slc"].mode = "pan_zoom"
            yield
            self.vwr.layers["spk_slc"].mode = "paint"

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
                self.vwr.layers["spk_slc"].mode = "erase"
                yield
                self.vwr.layers["spk_slc"].mode = "paint"
        
#%% Class(AnnotateSPK) : init_layers() ----------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.asc_nrm, name="asc_nrm", colormap="gray", visible=1,
            blending="minimum", opacity=0.75, gamma=0.5, 
            )
        
        self.vwr.add_image(
            self.spk_msk, name="spk_msk", colormap="yellow", visible=1,
            blending="additive", opacity=0.25,
            )
        
        self.vwr.add_labels(
            self.spk_slc, name="spk_slc", visible=1,
            blending="translucent", opacity=0.75, 
            )
        
        self.vwr.layers["spk_slc"].selected_label = 1
        self.vwr.layers["spk_slc"].brush_size = 10
        self.vwr.layers["spk_slc"].mode = "paint"
        self.get_info()

#%% Class(Classify) : info() --------------------------------------------------

    def get_info(self):

        n_add = self.spk_cls["c"].count(1)
        n_rmv = self.spk_cls["c"].count(2)

        self.info.setText(
            f"{self.vws_paths[self.vws_idx].name}\n"
            f"n valid   = {n_add}\n"
            f"n invalid = {n_rmv}"
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    annSPK = AnnotateSPK(main)
    spk_cls = annSPK.spk_cls
