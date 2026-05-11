#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from pathlib import Path

# functions
from functions import (
    get_crops, save_compressed_stack, load_compressed_stack)

# bdtools
from bdtools.model.model import Model

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    data_path = Path("D:\local_mbl\data")
    vws_path = data_path / "251126_D7_S2_Scan1_unmixed_00"
    ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
    stk_nrm_path = vws_path / "stack_norm.npz"
    model_aec_path = Path.cwd() / "model-aec_32_3_128_59273-None"
    
    # Parameters
    crop_size = 32
    crop_fraction = 1
    stk_level = 1
    
    # Load data
    stk_nrm = load_compressed_stack(stk_nrm_path)
    with open(ncl_ctd_path, "rb") as file:
        ncl_ctd = pickle.load(file)
    
    # Get crops
    crp_nrm = get_crops(
        stk_nrm, ncl_ctd, 
        crop_size=crop_size, 
        crop_fraction=crop_fraction, 
        stk_level=stk_level,
        )
    
    # Prepare crops
    crp_nrm = crp_nrm[:, :7, ...] # remove autofluo channel
    crp_nrm = np.transpose(crp_nrm, (0, 2, 3, 1)) # transposition
    crp_nrm = crp_nrm.astype("float32") / 255
    
    # Load model
    model_aec = Model(parameters=None, model_path=model_aec_path)
    
    # Predict (latent space)
    lsp = model_aec.predict(
        crp_nrm, 
        patch_overlap=0, 
        batch_size=32, 
        chunk_size=10000, 
        latent=True
        )
    
#%% Classification DBSCAN -----------------------------------------------------

    # import matplotlib.pyplot as plt
    # from sklearn.cluster import DBSCAN
    # from sklearn.decomposition import PCA
    # from sklearn.neighbors import NearestNeighbors
    # from sklearn.preprocessing import StandardScaler

    # # Parameters
    # eps = 8.0
    # min_samples = 50
    # leaf_size = 30
    
    # # Scaling
    # scaler = StandardScaler()
    # lspp = scaler.fit_transform(lsp)
    # # lspp = lsp.copy()
    
    # # PCA
    # pca = PCA(n_components=32, random_state=42)
    # lspp = pca.fit_transform(lspp)
    
    # -------------------------------------------------------------------------
    
    # # Finding optimal eps
    # nn = NearestNeighbors(n_neighbors=min_samples)
    # nn_fit = nn.fit(lspp)
    # distances, indices = nn_fit.kneighbors(lspp)
    
    # # Plot
    # distances = np.sort(distances[:, 4], axis=0)
    # plt.plot(distances)
    # plt.xlabel("Points sorted by distance")
    # plt.ylabel("f{min_samples}-Nearest Neighbor Distance")
    # plt.title("K-Distance Graph to find Optimal Eps")
    # plt.show()
    
    # -------------------------------------------------------------------------
    
    # # Initialize DBSCAN
    # dbscan = DBSCAN(
    #     eps=eps, 
    #     min_samples=min_samples, 
    #     leaf_size=leaf_size,
    #     n_jobs=-1,
    #     )
    
    # # Fit & predict
    # t0 = time.time()
    # print("Fit & predict DBSCAN : ", end="", flush=True)
    # labels = dbscan.fit_predict(lspp)      
    # print(np.unique(labels))
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
        
#%% Classification HDBSCAN ----------------------------------------------------

    from sklearn.cluster import HDBSCAN 
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
        
    # Scaling
    scaler = StandardScaler()
    lspp = scaler.fit_transform(lsp)
    # lspp = lsp.copy()
    
    # PCA
    pca = PCA(n_components=32, random_state=42)
    lspp = pca.fit_transform(lspp)

    # Initialize HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=100, 
        min_samples=20, 
        leaf_size=250, 
        n_jobs=-1,
        )
    
    # Fit & predict
    t0 = time.time()
    print("Fit & predict HDBSCAN : ", end="", flush=True)
    labels = hdbscan.fit_predict(lspp)
    print(np.unique(labels))
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")  

#%% Classification HDBSCAN (old) ----------------------------------------------

    # from sklearn.cluster import HDBSCAN
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler

    # # Scaling
    # scaler = StandardScaler()
    # lspp = scaler.fit_transform(lsp)
    # # lspp = lsp.copy()
    
    # # PCA ---------------------------------------------------------------------
    
    # # 
    # t0 = time.time()
    # print("PCA : ", end="", flush=False)
    
    # pca = PCA(n_components=24)
    # lspp = pca.fit_transform(lspp)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # HDBSCAN -----------------------------------------------------------------
    
    # # 
    # t0 = time.time()
    # print("HDBSCAN : ", end="", flush=False)
    
    # hdbscan = HDBSCAN(
    #     min_cluster_size=100, 
    #     min_samples=20, 
    #     n_jobs=-1
    #     )
    # labels = hdbscan.fit_predict(lspp) 
    # print(np.unique(labels))    
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
#%% Display -------------------------------------------------------------------
    
    import napari
    from bdtools.norm import norm_pct
    
    labels = norm_pct(labels, pct_low=0, pct_high=100)   
    coords = np.column_stack((ncl_ctd["y"][1], ncl_ctd["x"][1])).astype(int)
    properties = {"cluster_id" : labels}
    
    vwr = napari.Viewer()
    vwr.add_image(stk_nrm)
    points_layer = vwr.add_points(
        coords,
        name="points",
        properties=properties,
        size=5,
        face_color="cluster_id",
        face_colormap="viridis",
        border_width=0.0,
        )
