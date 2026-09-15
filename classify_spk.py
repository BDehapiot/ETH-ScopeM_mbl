#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton, QWidget, QLabel,
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
        
#%% Class(Classify) : function(s) ---------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        stk_nrm_path = vws_path / "stack_norm.tif"
        asc_msk_path = vws_path / "asc_mask.tif"
        
        # Load data
        self.asc_nrm = io.imread(stk_nrm_path)[4, ...]
        self.asc_msk = io.imread(asc_msk_path)
        self.spk_msk = self.asc_msk == 200
        self.lbl_msk = np.zeros_like(self.asc_msk)
        
    def update_view(self):
        self.load_data()
        self.vwr.layers["asc_nrm"].data = self.asc_nrm
        self.vwr.layers["spk_msk"].data = self.spk_msk
        self.vwr.layers["lbl_msk"].data = self.lbl_msk
        
    def update_mask(self):
        spk_msk = self.vwr.layers["spk_msk"].data
        lbl_msk = self.vwr.layers["lbl_msk"].data
        lbl_msk[~spk_msk] = 0 
        self.vwr.layers["lbl_msk"].data = lbl_msk

    # Buttons -----------------------------------------------------------------
    
    def prev_view(self):
        if self.vws_idx > 0:
            self.vws_idx -= 1
        self.update_view()
        
    def next_view(self):
        if self.vws_idx < (len(self.vws_paths) - 1):
            self.vws_idx += 1
        self.update_view()
        
    def add_spk(self):
        self.vwr.layers["lbl_msk"].selected_label = 1 
    
    def remove_spk(self):
        self.vwr.layers["lbl_msk"].selected_label = 2 
        
    # Shorcuts ----------------------------------------------------------------
        
    def hide_mask(self):
        self.vwr.layers["spk_msk"].visible = 0
        self.vwr.layers["lbl_msk"].visible = 0
        
    def show_mask(self):
        self.vwr.layers["spk_msk"].visible = 1
        self.vwr.layers["lbl_msk"].visible = 1
        
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
        self.cls_add = QRadioButton("is valid")
        self.cls_rmv = QRadioButton("is not valid")
        self.cls_add.setChecked(True)
        self.cls_group_layout.addWidget(self.cls_add)
        self.cls_group_layout.addWidget(self.cls_rmv)
        self.cls_group_box.setLayout(self.cls_group_layout)
        
        # Create "Segmentation" menu
        self.seg_group_box = QGroupBox("Segmentation")
        seg_group_layout = QHBoxLayout()
        self.rad_semantic = QRadioButton("Semantic")
        self.rad_instance = QRadioButton("Instance")
        self.rad_semantic.setChecked(True)
        seg_group_layout.addWidget(self.rad_semantic)
        seg_group_layout.addWidget(self.rad_instance)
        self.seg_group_box.setLayout(seg_group_layout)
        
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
        self.cls_rmv.toggled.connect(self.remove_spk)
        
#%% Class(Classify) : shortcuts -----------------------------------------------   

        @self.vwr.bind_key("Delete", overwrite=True)
        def hide_labels_key(viewer):
            self.hide_mask()
            yield
            self.show_mask()
            
        @Labels.bind_key("Enter", overwrite=True)
        def save_mask_key(viewer):
            self.update_mask() 
        
#%% Class(Classify) : init_layers() -------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.asc_nrm, name="asc_nrm", colormap="gray", visible=1,
            opacity=1.0, gamma=0.4, 
            )
        
        self.vwr.add_image(
            self.spk_msk, name="spk_msk", colormap="blue", visible=1,
            blending="additive", opacity=0.5, gamma=1.0, 
            )
        
        self.vwr.add_labels(
            self.lbl_msk, name="lbl_msk", visible=1,
            blending="translucent", opacity=0.75, 
            )
        
        self.vwr.layers["lbl_msk"].brush_size = 20
        self.vwr.layers["lbl_msk"].mode = 'paint'

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    classify = Classify(main)