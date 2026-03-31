#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import get_crops

# tensorflow
from tensorflow.keras import layers, Model

#%% Inputs --------------------------------------------------------------------

vae_parameters = {
    
    # extract()    
    "n_crops"     : 2e4,
    "crop_size"   : 32,
    
    # build()
    "n_channels"  : 8,
    "latent_size" : 64,
    
    # train()
    "batch_size"  : 256,
    
    }

#%% Class(TrainVAE) -----------------------------------------------------------

class TrainVAE:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.extract()
        # self.build()
        
#%% Class(TrainVAE) : extract() -----------------------------------------------

    def extract(self):
        
        msk_name = f"mask_level-{main.process_level}.tif"
        stk_name = f"stack_level-{main.process_level}.tif"
        msk_paths = list(main.data_path.rglob(msk_name))
        stk_paths = list(main.data_path.rglob(stk_name))
        
        self.crops = []
        for p, (msk_path, stk_path) in enumerate(zip(msk_paths, stk_paths)):
            if p == 0:
                msk = io.imread(msk_path)
                stk = io.imread(stk_path)
                crops = get_crops(stk, msk, crop_size=32)
                self.idxs = np.random.choice(
                    crops.shape[0] + 1, size=int(self.n_crops), replace=False)
                # self.crops.append(crops[idxs, ...])
        
        
        
        pass
        
#%% Class(TrainVAE) : build() -------------------------------------------------
    
    def build(self):
            
        # Encoder -------------------------------------------------------------
        
        encoder_inputs = layers.Input(
            shape=(self.crop_size, self.crop_size, self.n_channels))
        
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        
        x = layers.Flatten()(x)
        latent_space = layers.Dense(
            self.latent_size, activation="relu", name="latent_features")(x)
        
        self.encoder = Model(encoder_inputs, latent_space, name="encoder")
    
        # Decoder -------------------------------------------------------------
    
        decoder_inputs = layers.Input(shape=(self.latent_size,))
    
        spatial_dim = self.crop_size // 4
        
        x = layers.Dense(spatial_dim * spatial_dim * 64, activation="relu")(decoder_inputs)
        x = layers.Reshape((spatial_dim, spatial_dim, 64))(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        decoder_outputs = layers.Conv2D(
            self.n_channels, (3, 3), activation="sigmoid", padding="same")(x)
        
        self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")
    
        # Full autoencoder ----------------------------------------------------
        
        autoencoder_inputs = layers.Input(shape=(
            self.crop_size, self.crop_size, self.n_channels))
        encoded_img = self.encoder(autoencoder_inputs)
        reconstructed_img = self.decoder(encoded_img)
        
        self.autoencoder = Model(
            autoencoder_inputs, reconstructed_img, name="autoencoder")
        self.autoencoder.compile(optimizer="adam", loss="mse")
    
#%% Class(TrainVAE) : train() -------------------------------------------------

    def train(self):
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainVAE(main, parameters=vae_parameters)
    
    idxs = train.idxs