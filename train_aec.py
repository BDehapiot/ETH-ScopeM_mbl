#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from main import Main, main_parameters

# functions
from functions import (
    norm_data, get_crops, 
    save_compressed_stack, load_compressed_stack
    )

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

#%% Inputs --------------------------------------------------------------------

aec_parameters = {
    
    # Procedure
    "run_crops"     : 0,
    "run_train"     : 1,
    
    # get_crops() 
    "crop_size"     : 32,
    "crop_fraction" : 0.01,
    "stk_level"     : 1,
    
    # build()
    "n_channels"    : 8,
    "latent_size"   : 64,
    
    # # train()
    # "batch_size"    : 256,
    
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
        
        crp_nrm = []
        for p, vws_path in enumerate(self.main.vws_paths):
            
            print(f"Main() : get_crops() : {vws_path.name}")
            
            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"
            crp_nrm_path = self.main.data_path / "crops_raw.npz"
            
            # Load data
            with open(ncl_ctd_path, "rb") as file:
                ncl_ctd = pickle.load(file)
            stk_nrm = load_compressed_stack(stk_nrm_path)

            # Get crops
            crp_nrm.append(get_crops(
                stk_nrm, ncl_ctd, 
                crop_size=self.crop_size, 
                crop_fraction=self.crop_fraction, 
                stk_level=self.stk_level,
                ))
            
        # Save
        crp_nrm = np.concatenate(crp_nrm, axis=0)
        save_compressed_stack(crp_nrm, crp_nrm_path)
        
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
    
#%% Class(TrainAEC) : train() -------------------------------------------------

    def train(self):
        
        # Paths
        self.crp_nrm_path = self.main.data_path / "crops_raw.npz"
        
        # Load data
        self.crp_nrm = load_compressed_stack(self.crp_nrm_path)
        self.crp_nrm = self.crp_nrm.astype("float32") / 255
        
        # Split data
        
        
        # crop_idx = np.random.choice(
        #     np.arange(0, len(ys)), 
        #     size=int(len(ys) * crop_fraction), 
        #     replace=False,
        #     )
        
            
        # # 2. SHUFFLE AND SPLIT DATA (80% Train, 20% Validation)
        # dataset_size = len(self.crops)
        # val_size = int(0.2 * dataset_size)
        # train_size = dataset_size - val_size
        
        # dataset = tf.data.Dataset.from_tensor_slices((self.crops, self.crops))
        # dataset = dataset.shuffle(buffer_size=1024) # Shuffle before splitting
        
        # train_dataset = dataset.take(train_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        # val_dataset = dataset.skip(train_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # # 3. CONFIGURE CALLBACKS
        # callbacks = [
        #     # Stop training if validation loss doesn't improve for 5 epochs
        #     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        #     # Reduce learning rate if training plateaus
        #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        #     # Save the best overall autoencoder model automatically
        #     ModelCheckpoint(
        #         filepath=str(self.main.data_path / "autoencoder_best.keras"), 
        #         save_best_only=True, 
        #         monitor='val_loss'
        #     )
        # ]
        
        # # 4. TRAIN THE MODEL
        # # We can safely increase epochs now because EarlyStopping will halt it if needed
        # self.history = self.autoencoder.fit(
        #     train_dataset, 
        #     validation_data=val_dataset,
        #     epochs=50, 
        #     callbacks=callbacks
        # )
        
        # # 5. SAVE THE ENCODER (Usually the most useful part for future predictions)
        # # We save it in the newer .keras format (recommended over .h5)
        # self.encoder.save(self.main.data_path / "encoder.keras")
        # print("Training complete. Models saved!")
    
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(main, parameters=aec_parameters)
    crp_nrm = train.crp_nrm