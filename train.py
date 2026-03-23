#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# bdtools
from bdtools.unet.unet import UNet
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

train_parameters = {
    
    # Procedure
    "run_patches"  : 0,
    "run_annotate" : 0,
    "run_train"    : 1,
    
    # Paths
    "data_path"  : Path("D:\local_mbl\data"),
    "train_path" : Path.cwd() / "data" / "train", 
    
    # extract_patches()
    "n_patches"             : 20, # save n_patches per DAPI image
    "extract_patch_size"    : 256,
    "extract_patch_overlap" : 128,
    
    # train() -----------------------------------------------------------------
    
    # Paths
    "root_path"          : None,
    "model_name"         : "model_256_edt_512-28",

    # Build
    "input_shape"        : (None, None, 1),
    "backbone"           : "resnet18",
    "activation"         : "sigmoid",
        
    # Train
    "epochs"             : 256,
    "batch_size"         : 16,
    "validation_split"   : 0.2,
    "metric"             : "soft_dice_coef",
    "learning_rate"      : 0.001,
    "patience"           : 64,

    # Prepare
    "patch_size"         : 256,
    "patch_overlap"      : 0,
    "mask_method"        : "edt",
            
    # Augment
    "augment_iterations" : 512,
    "augment_invert_p"   : 0,
    "augment_gamma_p"    : 0.5,
    "augment_gblur_p"    : 0.5,
    "augment_noise_p"    : 0.5,
    "augment_flip_p"     : 0.5,
    "augment_distord_p"  : 0.5,
    
    }

#%% Class(Train) --------------------------------------------------------------

class Train:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_patches:
            self.extract_patches()
        if self.run_annotate:
            self.annotate()
        if self.run_train:
            self.train()

#%% Class(Train) extract_patches() --------------------------------------------

    def extract_patches(self):
        
        for p, path in enumerate(self.main.dapi_paths):
            dapi = io.imread(path)
            patches = extract_patches(
                dapi, self.extract_patch_size, self.extract_patch_overlap)
            idxs =  np.random.choice(
                np.arange(0, len(patches)), size=self.n_patches, replace=False)
            for idx in idxs:
                patch_name = (
                    f"{str(path.parent.name)}_"
                    f"{path.stem}_"
                    f"{idx:04d}.tif"
                    )
                io.imsave(
                    self.train_path / patch_name,
                    patches[idx], check_contrast=False,
                    )
    
#%% Class(Train) annotate() ---------------------------------------------------

    def annotate(self):
        Annotate(self.train_path)  
        
#%% Class(Train) train() ------------------------------------------------------

    def train(self):
        
        # Load data
        msk_paths = list(self.train_path.glob("*mask.tif"))
        msks, imgs = [], []
        for path in msk_paths:
            msks.append(io.imread(path))
            imgs.append(io.imread(
                str(path).replace("_mask", "")
                ))
        msks = np.stack(msks)
        imgs = np.stack(imgs)
            
        # Normalize data (images)
        imgs = imgs.astype("float32") / 255
            
        # Setup model
        self.unet = UNet(parameters=self.parameters, model_path=None)
        
        # # Train model
        # self.unet.train(imgs, msks)
            
        self.msks = msks
        self.imgs = imgs
        
#%% Class(Predict) predict() --------------------------------------------------

    def predict(self, X):
        pass
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = Train(main, parameters=train_parameters)
    
    # -------------------------------------------------------------------------
        
    dapi_path = Path("D:\local_mbl\data\\251126_D7_S2\dapi_00.tif")
    dapi = io.imread(dapi_path).squeeze()
    dapi = dapi.astype("float32") / 255
    prd = train.unet.predict(dapi)
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(dapi)
    vwr.add_image(prd, blending="additive", colormap="inferno")
    
    # msks = train.msks    
    # imgs = train.imgs
    # X_trn = train.unet.X_trn
    # y_trn = train.unet.y_trn
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(np.stack(X_val))
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(np.stack(imgs))
    # vwr.add_image(np.stack(msks))