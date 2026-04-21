#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import get_crops

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model

#%% Inputs --------------------------------------------------------------------

aec_parameters = {
    
    # Procedure
    "run_crops"   : 1,
    "run_train"   : 0,
    
    # extract()    
    "n_crops"     : 5000,
    "crop_size"   : 32,
    
    # build()
    "n_channels"  : 8,
    "latent_size" : 64,
    
    # train()
    "batch_size"  : 256,
    
    }

#%% Class(TrainAEC) -----------------------------------------------------------

class TrainAEC:
    def __init__(self, main, parameters=None):
        self.main = main
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_crops:
            self.extract_crops()
        if self.run_train:
            self.build()
            self.train()
        
#%% Class(TrainAEC) : extract() -----------------------------------------------

    def extract_crops(self):
        
        # Paths
        stk_name = f"stack_level-{main.process_level}.tif"
        msk_name = f"mask_level-{main.process_level}.tif"
        prp_name = f"props_level-{main.process_level}.pkl"
        stk_paths = list(self.main.data_path.rglob(stk_name))
        msk_paths = list(self.main.data_path.rglob(msk_name))
        prp_paths = list(self.main.data_path.rglob(prp_name))
        crop_path = self.main.data_path / "crops.tif"
        
        # Extract crops
        self.crops = []
        for p, (stk_path, msk_path, prp_path) in enumerate(zip(
                stk_paths, msk_paths, prp_paths)):
            print(
                f"TrainAEC() : extract() : "
                f"{msk_path.parent.parent.name}/{msk_path.parent.name}"
                )
            stk = io.imread(stk_path)
            msk = io.imread(msk_path)
            with open(prp_path, "rb") as file:
                prp = pickle.load(file)
            crops = get_crops(stk, msk, prp=prp, crop_size=32)
            self.idxs = np.random.choice(
                crops.shape[0], size=self.n_crops, replace=False)
            self.crops.append(crops[self.idxs, ...])
        self.crops = np.concatenate(self.crops, axis=0)
        
        # Save crops
        io.imsave(crop_path, self.crops, check_contrast=False)
        
#%% Class(TrainAEC) : build() -------------------------------------------------
    
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
    
#%% Class(Trainaec) : train() -------------------------------------------------

    def train(self):
        
        # Load crops (if necessary)
        if not hasattr(self, "crops"):
            crop_path = self.main.data_path / "crops.tif"
            self.crops = io.imread(crop_path)
        
        dataset = tf.data.Dataset.from_tensor_slices((self.crops, self.crops))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.history = self.autoencoder.fit(dataset, epochs=20)
        
        # self.latent_vectors = self.encoder.predict(self.crops, batch_size=self.batch_size)
        

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(main, parameters=aec_parameters)
    