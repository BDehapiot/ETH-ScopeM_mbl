#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path
from main import Main, main_parameters

# functions
from functions import (
    get_crops, save_compressed_stack, load_compressed_stack)

# bdtools
from bdtools.unet import metrics

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#%% Inputs --------------------------------------------------------------------

aec_parameters = {
    
    # Procedure
    "run_crops"        : 0,
    "run_train"        : 0,
        
    # get_crops() 
    "crop_size"        : 32,
    "crop_fraction"    : 0.01,
    "stk_level"        : 1,
    
    # build()
    "n_channels"       : 7,
    "latent_size"      : 512,
    
    # train()
    "epochs"           : 256,
    "batch_size"       : 256,
    "validation_split" : 0.2,
    "metric"           : "soft_dice_coef",
    "learning_rate"    : 0.001,
    "patience"         : 64,
        
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
        
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            dice_loss = 1 - metrics.soft_dice_coef(y_true, y_pred)
            return bce + (5.0 * dice_loss)
        
        # Encoder -------------------------------------------------------------
        
        encoder_inputs = layers.Input(
            shape=(self.crop_size, self.crop_size, self.n_channels))
        
        x = layers.Conv2D(32, (3, 3), padding="same")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        latent_space = layers.Dense(
            self.latent_size, activation="relu", name="latent_features")(x)
        
        self.encoder = Model(encoder_inputs, latent_space, name="encoder")
    
        # Decoder -------------------------------------------------------------
    
        decoder_inputs = layers.Input(shape=(self.latent_size,))
    
        spatial_dim = self.crop_size // 4
        
        x = layers.Dense(spatial_dim * spatial_dim * 64, activation="relu")(decoder_inputs)
        x = layers.Reshape((spatial_dim, spatial_dim, 64))(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
                
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
        
        # Compile        
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            # loss="binary_crossentropy", # Parameter
            loss=combined_loss,
            metrics=[getattr(metrics, self.metric)],
            )
        
#%% Class(TrainAEC) : train() -------------------------------------------------

    def train(self):
        
        # Paths
        self.crp_nrm_path = self.main.data_path / "crops_raw.npz"

        # Load data
        self.crp_nrm = load_compressed_stack(self.crp_nrm_path)
        self.crp_nrm = self.crp_nrm.astype("float32") / 255
        self.crp_nrm = self.crp_nrm[:, :7, ...] # remove autofluo channel
        
        # Create model directory
        model_name = (
            "model-aec_"
            f"{self.crop_size}_"
            f"{self.latent_size}_"
            f"{self.crp_nrm.shape[0]}"
            )
        model_path = Path.cwd() / model_name
        if not model_path.exists():
            model_path.mkdir(exist_ok=True)
        
        # Split data
        crp_num = self.crp_nrm.shape[0]
        trn_num = int(crp_num * (1 - self.validation_split))
        idxs = np.random.choice(
            np.arange(0, crp_num), size=crp_num, replace=False)
        idx_trn = idxs[0:trn_num]
        idx_val = idxs[trn_num:]
        X_trn = self.crp_nrm[idx_trn]
        X_val = self.crp_nrm[idx_val]
        
        # Transpose
        X_trn = np.transpose(X_trn, (0, 2, 3, 1))
        X_val = np.transpose(X_val, (0, 2, 3, 1))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", 
                patience=self.patience, 
                restore_best_weights=True
                ),
            ModelCheckpoint(
                filepath=model_path / "weights.keras", 
                save_best_only=True, 
                monitor="val_loss"
                ),
            ]
        
        # Train
        bs = self.batch_size
        trn_dataset = tf.data.Dataset.from_tensor_slices((X_trn, X_trn))
        trn_dataset = trn_dataset.batch(bs).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val))
        val_dataset = val_dataset.batch(bs).prefetch(tf.data.AUTOTUNE)

        # Train    
        self.history = self.autoencoder.fit(
            x=trn_dataset,
            validation_data=val_dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            )
        
#%% Class(TrainAEC) : predict() -------------------------------------------------

    def predict(self, data, batch_size=None, decoder=False):
            
        # Paths
        weights_path = self.main.data_path / "weights.keras"
               
        # Build
        self.build()
        self.autoencoder.load_weights(str(weights_path))
        
        # Format data
        X = data.astype("float32") / 255.0
        X = np.transpose(X, (0, 2, 3, 1))
        
        # Predict
        if batch_size is None:
            batch_size = self.batch_size
        if decoder:
            prd = self.autoencoder.predict(X, batch_size=batch_size)
            prd = np.transpose(prd, (0, 3, 1, 2))
        else:
            prd = self.encoder.predict(X, batch_size=batch_size)
            
        return prd
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    train = TrainAEC(main, parameters=aec_parameters)
    
#%% Predict
    
    # Parameters
    subset = 0.01

    # Paths
    crp_nrm_path = main.data_path / "crops_raw.npz"
    
    # Load
    crp_nrm = load_compressed_stack(crp_nrm_path)
    crp_num = crp_nrm.shape[0]   
    
    # Select data
    idxs = np.random.choice(
        np.arange(0, crp_num), size=int(crp_num * subset), replace=False)
    X = crp_nrm[idxs][:, :7, ...]
    
    # Predict
    prd = train.predict(X)
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(X)
    vwr.add_image(prd)