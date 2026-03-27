#%% Imports -------------------------------------------------------------------

import gc
import pickle
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    load_metadata, load_arr,
    )

# bdtools
from bdtools.norm import norm_pct
from bdtools.conn import lbl_conn
from bdtools.unet.unet import UNet

# skimage
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops_table
from skimage.segmentation import watershed, expand_labels

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Channels
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF"
        ],
    
    # Paths
    "data_path"       : Path("D:\local_mbl\data"),
    "load_model_name" : "model_256_edt_1024-52",

    # extract()
    "preview_level"   : 4,
    
    # process()
    "process_level"   : 1,
    "thresh_0"        : 0.25,
    "thresh_1"        : 0.50,
        
    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.initialize()
        self.extract()
        self.process()
        # self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        self.qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))
        self.sld_paths = [
            self.data_path / f"{path.stem[:12]}"
            for path in self.qpt_paths
            ]
       
        # Setup slide directory
        for sld_path in self.sld_paths:
            if not sld_path.exists():
                sld_path.mkdir(parents=True, exist_ok=True) 

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        self.vws_paths = [] 
        for p, path in enumerate(self.qpt_paths): 
            sld_path = self.sld_paths[p]
            mtd_path = sld_path / "metadata.pkl"
            pvw_path = sld_path / f"preview_level-{self.preview_level}.tif"
            
            # Metadata
            if not mtd_path.exists():
                print(f"extract() : {path.name}")
                mtd = load_metadata(path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
        
            # Preview
            if not pvw_path.exists():
                pvw = load_arr(
                    path, metadata=mtd, 
                    channels="all", level=self.preview_level
                    )
                pvw = np.mean(pvw, axis=0)
                pvw = (norm_pct(pvw) * 255).astype("uint8")
                io.imsave(pvw_path, pvw, check_contrast=False)

            # View(s)
            for idx in range(len(mtd["roi"])):
                
                # Setup view directory
                vws_path = sld_path / f"view_{idx:02d}"
                self.vws_paths.append(vws_path)
                if not vws_path.exists():
                    vws_path.mkdir(parents=True, exist_ok=True) 
                
                # DAPI at level 0 (for predictions)
                dap_path = vws_path / "dapi_level-0.tif"
                if not dap_path.exists():
                    dap = load_arr(
                        path, metadata=mtd, 
                        channels="DAPI", level=0, roi_idx=idx,
                        )
                    dap = (norm_pct(dap) * 255).astype("uint8")
                    io.imsave(dap_path, dap, check_contrast=False)
                    
                # Stack at process level 
                stk_path = vws_path / f"stack_level-{self.process_level}.tif"
                if not stk_path.exists():
                    stk = load_arr(
                        path, metadata=mtd, 
                        channels="all", level=self.process_level, roi_idx=idx,
                        )
                    stk = ((stk + 8) / 16 ).astype("uint8") # 12bits to 8bits
                    io.imsave(stk_path, stk, check_contrast=False)                               
                        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):

        for p, path in enumerate(self.vws_paths):
            dap_path = path / "dapi_level-0.tif"
            msk_path = path / f"mask_level-{self.process_level}.tif"

            if not msk_path.exists():
                print(f"process() : {path.parent.name}/{path.name}")
                
                if "unet" not in locals():
                    unet = UNet(
                        parameters=None, 
                        model_path=Path.cwd() / self.load_model_name
                        )
                
                dap = io.imread(dap_path)
                dap = dap.astype("float32") / 255
            
                # Predict
                prd = unet.predict(dap)
                dscale_factor = 2 ** self.process_level
                prd = downscale_local_mean(prd, dscale_factor)
                
                # Get mask
                msk_0 = prd > self.thresh_0
                msk_1 = prd > self.thresh_1
                msk = watershed(-prd, label(msk_1), mask=msk_0)
                msk[lbl_conn(msk, conn=2) == 2] = 0
                msk = remove_small_objects(
                    msk, min_size=0.5 * (2 ** self.process_level) ** 2)
                msk = ((msk > 0) * 255).astype("uint8")
                              
                # Save & clear
                io.imsave(msk_path, msk, check_contrast=False)
                tf.keras.backend.clear_session()
                gc.collect()
            
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        for p, path in enumerate(self.view_paths):
            msk_path = path / f"mask_level-{self.process_level}.tif"
            stk_path = path / f"stack_level-{self.process_level}.tif"
            results_path = path / f"results_level-{self.process_level}.csv"
            
            if not results_path.exists():
                print(f"analyse() : {path.parent.name}/{path.name}")
                
                msk = io.imread(msk_path) > 0
                stk = io.imread(stk_path)
                
                # Measure intensities
                msk = expand_labels(
                    label(msk), distance=1 * (2 ** self.process_level) ** 2)
                results = regionprops_table(
                    msk, intensity_image=np.moveaxis(stk, 0, -1), 
                    properties=("label", "intensity_mean"),
                    )
                results.pop("label", None)
                results = {
                    channel : value 
                    for channel, value 
                    in zip(self.channels, results.values())
                    }
                results = pd.DataFrame(results)
                results.to_csv(results_path, index=False)
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)

#%%

    import time
    
    # -------------------------------------------------------------------------

    view_idx = 0
    channels = main_parameters["channels"]
    distance = 3 * (2 ** main.process_level) ** 2
    msk_path = main.vws_paths[view_idx] / f"mask_level-{main.process_level}.tif"
    stk_path = main.vws_paths[view_idx] / f"stack_level-{main.process_level}.tif"
    msk = io.imread(msk_path)
    stk = io.imread(stk_path)
    
    # -------------------------------------------------------------------------

    t0 = time.time()
    print("regionprops_table() : ", end="", flush=False)

    results = regionprops_table(
        label(msk), properties=("label", "centroid"),
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    results["y"] = results.pop("centroid-0")
    results["x"] = results.pop("centroid-1")
    
#%%
            
    def get_crops(stk, y_coords, x_coords, crop_size=32):

        y_coords = y_coords.astype("int32")
        x_coords = x_coords.astype("int32")
        
        # Normalize (0 to 1)
        if np.issubdtype(stk.dtype, np.integer):
            for i, img in enumerate(stk):
                stk[i] = norm_pct(img)
        
        # Pad (acc. to crop half size)
        chs = crop_size // 2 # crop half size
        pad_width = ((0, 0), (chs, chs), (chs, chs))
        stk = np.pad(stk, pad_width=pad_width, mode="reflect")

        # Extract crops
        crops = []
        for (y, x) in zip(y_coords, x_coords):
            y0 = y; y1 = y + crop_size
            x0 = x; x1 = x + crop_size
            if y1 - y0 != crop_size:
                print(y0, y1)
            if x1 - x0 != crop_size:
                print(x0, x1)
            crops.append(
                np.swapaxes(stk[..., y0:y1, x0:x1], 0, -1)
                )
            
        return np.stack(crops)
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("get_crops() : ", end="", flush=False)
    
    crops = get_crops(stk, results["y"], results["x"], crop_size=32)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
        
#%%

    import tensorflow as tf
    from tensorflow.keras import layers, Model
        
    def build_autoencoder(crop_size, n_channels, latent_dim=64):
        """
        Builds and compiles a Convolutional Autoencoder.
        
        Returns:
            encoder: Model used to compress images into latent vectors.
            decoder: Model used to reconstruct images from latent vectors.
            autoencoder: The full compiled model for training.
        """
        
        # ==========================================
        # 1. ENCODER
        # ==========================================
        encoder_inputs = layers.Input(shape=(crop_size, crop_size, n_channels))
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Flatten()(x)
        latent_space = layers.Dense(latent_dim, activation='relu', name='latent_features')(x)
        
        encoder = Model(encoder_inputs, latent_space, name='encoder')
    
        # ==========================================
        # 2. DECODER
        # ==========================================
        decoder_inputs = layers.Input(shape=(latent_dim,))
        
        # Calculate the spatial dimensions before flattening
        # Since we max-pooled twice (divided by 2 twice), the size is crop_size // 4
        spatial_dim = crop_size // 4
        
        x = layers.Dense(spatial_dim * spatial_dim * 64, activation='relu')(decoder_inputs)
        x = layers.Reshape((spatial_dim, spatial_dim, 64))(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        decoder_outputs = layers.Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')(x)
        
        decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
    
        # ==========================================
        # 3. FULL AUTOENCODER
        # ==========================================
        autoencoder_inputs = layers.Input(shape=(crop_size, crop_size, n_channels))
        encoded_img = encoder(autoencoder_inputs)
        reconstructed_img = decoder(encoded_img)
        
        autoencoder = Model(autoencoder_inputs, reconstructed_img, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return encoder, decoder, autoencoder

#%% 

    crop_size  = 32      # Assuming 32x32 crops (y, x)
    n_channels = 8       # Replace with your actual number of channels
    latent_dim = 64      # The size of the feature vector you'll cluster later

    encoder, decoder, autoencoder = build_autoencoder(
        crop_size=crop_size, n_channels=n_channels, latent_dim=latent_dim)
    
    # -------------------------------------------------------------------------
    
    batch_size = 256
    train_dataset = tf.data.Dataset.from_tensor_slices((crops, crops))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    history = autoencoder.fit(train_dataset, epochs=5)
    latent_vectors = encoder.predict(crops, batch_size=batch_size)
    
#%%
    
    idxs = np.random.choice(latent_vectors.shape[0], size=2000, replace=False)
    test = latent_vectors[idxs, :]
    
    import napari        
    vwr = napari.Viewer()
    vwr.add_image(test)
    
    
    