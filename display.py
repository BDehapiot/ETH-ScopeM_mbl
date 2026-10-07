#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import get_mask_from_coords

# Qt
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QRadioButton, QCheckBox, 
    QWidget, QLabel,
    )

#%% Class(Display) ------------------------------------------------------------

class Display:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = main.parameters
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
        
        clrs = self.ncl_colors
        clss = self.ncl_classes
        self.ncl_class_map = {
            "cls_oth" : {"tag" : 0, "color" : clrs[0], "class" : clss[0]},
            "cls_epi" : {"tag" : 1, "color" : clrs[1], "class" : clss[1]},
            "cls_neu" : {"tag" : 2, "color" : clrs[2], "class" : clss[2]},
            "cls_mac" : {"tag" : 3, "color" : clrs[3], "class" : clss[3]},
            "cls_bcl" : {"tag" : 4, "color" : clrs[4], "class" : clss[4]},
            "cls_tcl" : {"tag" : 5, "color" : clrs[5], "class" : clss[5]},
            }
        
        self.ncl_color_map = {
            v["tag"]: v["color"] for v in self.ncl_class_map.values()}
                
#%% Class(Display) : function(s) ----------------------------------------------

    # Logic & updates ---------------------------------------------------------
            
    def load_data(self):
        
        # Paths
        vws_path = self.vws_paths[self.vws_idx]
        stk_nrm_path = vws_path / "stack_norm.tif"
        ncl_prp_path = vws_path / "nuclei_properties.pkl"
        ncl_cls_path = vws_path / "nuclei_classes.pkl"
        # ncl_plt_path = vws_path / "nuclei_plot.pkl"
        asc_prp_path = vws_path / "asc_properties.pkl"
        asc_cls_path = vws_path / "asc_classes.pkl"
        # asc_plt_path = vws_path / "asc_plot.pkl"
        
        # Load
        with open(ncl_prp_path, "rb") as file:
            self.ncl_prp = pickle.load(file)
        with open(ncl_cls_path, "rb") as file:
            self.ncl_cls = pickle.load(file)["class"]
        with open(asc_prp_path, "rb") as file:
            self.asc_prp = pickle.load(file)
        with open(asc_cls_path, "rb") as file:
            self.asc_cls = pickle.load(file)["class"] + 1
        self.stk_nrm = io.imread(stk_nrm_path)
        
        # self.ncl_plt = QPixmap(str(ncl_plt_path))
        # self.ncl_plt = self.ncl_plt.scaledToWidth(
        #     int(self.dock_width / 1.75), Qt.SmoothTransformation)
        
        self.ncl_ctd = np.column_stack((
            self.ncl_prp["y"], self.ncl_prp["x"]))
        self.asc_msk = get_mask_from_coords(
            self.asc_prp["coords"], 
            labels=self.asc_cls,
            shape=self.stk_nrm[0, ...].shape
            )
        
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
        self.vwr.layers["asc_msk"].data = self.asc_msk
        # self.plt_label.setPixmap(self.ncl_plt)
        self.update_ncl_class()
        self.auto_contrast()
    
    def update_channel(self):    
        self.chn = self.chn_group.checkedButton().text()
        self.chn_idx = self.channels.index(self.chn)
        self.vwr.layers["stk_nrm"].data = self.stk_nrm[self.chn_idx]
        self.auto_contrast()
                
    def update_ncl_class(self):
        self.cll = [btn.text() for btn in self.ncl_group.buttons() if btn.isChecked()]
        self.cls_idx = [self.ncl_classes.index(c) for c in self.cll]
        self.ncl_msk = np.isin(self.ncl_cls, self.cls_idx)
        self.vwr.layers["ncl_ctd"].shown = self.ncl_msk
        
    def update_asc_class(self):
        if self.chk_asc.isChecked():
            self.vwr.layers["asc_msk"].visible = True
        else:
            self.vwr.layers["asc_msk"].visible = False
        
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
        
        # Create "Select view" menu
        self.vws_group_box = QGroupBox("Select view")
        self.vws_group_layout = QHBoxLayout()
        self.vws_prv = QPushButton("prev")
        self.vws_nxt = QPushButton("next")
        self.vws_group_layout.addWidget(self.vws_prv)
        self.vws_group_layout.addWidget(self.vws_nxt)
        self.vws_group_box.setLayout(self.vws_group_layout)
        self.vws_prv.clicked.connect(self.prev_view)
        self.vws_nxt.clicked.connect(self.next_view)
        
        # Create "Select channel" menu
        self.chn_group = QButtonGroup()
        self.chn_group_box = QGroupBox("Select channel")
        self.chn_group_layout = QHBoxLayout()
        for c in self.channels:
            setattr(self, f"rad_{c}", QRadioButton(f"{c}"))
            if c == self.channels[self.chn_idx]:
                getattr(self, f"rad_{c}").setChecked(True)
            self.chn_group.addButton(getattr(self, f"rad_{c}"))
            self.chn_group_layout.addWidget(getattr(self, f"rad_{c}"))
            getattr(self, f"rad_{c}").toggled.connect(self.update_channel)
        self.chn_group_box.setLayout(self.chn_group_layout)
               
        # Create "Check nuclei class" menu
        self.ncl_group = QButtonGroup()
        self.ncl_group.setExclusive(False)
        self.ncl_group_box = QGroupBox("Check nuclei class")
        self.ncl_group_layout = QHBoxLayout()
        for c in self.ncl_classes:
            setattr(self, f"chk_{c}", QCheckBox(f"{c}")) 
            if c == self.ncl_classes[self.cls_idx]:
                getattr(self, f"chk_{c}").setChecked(True)
            self.ncl_group.addButton(getattr(self, f"chk_{c}"))
            self.ncl_group_layout.addWidget(getattr(self, f"chk_{c}"))
            getattr(self, f"chk_{c}").stateChanged.connect(self.update_ncl_class)
        self.ncl_group_box.setLayout(self.ncl_group_layout)
        
        # Create "Check ASC class" menu
        self.asc_group = QButtonGroup()
        self.asc_group.setExclusive(False)
        self.asc_group_box = QGroupBox("Check ASC class")
        self.asc_group_layout = QHBoxLayout()
        self.chk_asc = QCheckBox("ASC")
        self.asc_group.addButton(self.chk_asc)
        self.asc_group_layout.addWidget(self.chk_asc)
        self.asc_group_box.setLayout(self.asc_group_layout)
        self.chk_asc.stateChanged.connect(self.update_asc_class)
        
        # Create "plot" menu
        self.plt_label = QLabel()
        self.plt_label.setAlignment(Qt.AlignCenter)
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vws_group_box)
        self.layout.addWidget(self.chn_group_box)
        self.layout.addWidget(self.ncl_group_box)
        self.layout.addWidget(self.asc_group_box)
        self.layout.addWidget(self.plt_label)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        self.dock_width = self.widget.width()
        
        # Shortcuts -----------------------------------------------------------

        @self.vwr.bind_key("PageDown", overwrite=True)
        def _prev_view(vwr):
            self.prev_view()
                    
        @self.vwr.bind_key("PageUp", overwrite=True)
        def _next_view(vwr):
            self.next_view()

#%% Class(Display) : init_layers() --------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        self.vwr.add_image(
            self.stk_nrm[self.chn_idx], name="stk_nrm", visible=1,
            blending="additive", opacity=0.6, gamma=0.4, 
            )
                        
        self.vwr.add_points(
            self.ncl_ctd, name="ncl_ctd", visible=1,
            blending="translucent", size=6, border_width=0.0,
            features={"class_label": self.ncl_cls},
            face_color={"colors": "class_label", "color_mode": "cycle"},
            face_color_cycle=self.ncl_color_map,
            border_color={"colors": "class_label", "color_mode": "cycle"},
            border_color_cycle=self.ncl_color_map,
            opacity=0.6, 
            )
        
        self.vwr.add_labels(
            self.asc_msk, name="asc_msk", visible=0,
            blending="additive", opacity=0.8, 
            )
        
        # self.plt_label.setPixmap(self.ncl_plt)
        self.update_ncl_class()
        self.auto_contrast()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    display = Display(main)
    napari.run()
    
    asc_cls = display.asc_cls