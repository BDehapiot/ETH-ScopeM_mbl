#%% Imports -------------------------------------------------------------------

import gc
import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
from pathlib import Path

# functions
from functions import (
    prep_preview, norm_12_to_8bits, norm_data, 
    get_slide_metadata, load_view, 
    get_nuclei_mask, get_nuclei_centroids, 
    save_compressed_stack, load_compressed_stack,
    save_compressed_mask, load_compressed_mask,
    get_crops, prepare_crops,
    get_preview_stats, format_preview_stats,
    get_nuclei_stats, format_nuclei_stats,
    plot,
    )

# bdtools
from bdtools.model.model import Model

# scipy
from scipy.stats import entropy

#%% Inputs --------------------------------------------------------------------

main_parameters = {
    
    # Maps
    "channels" : [
        "DAPI", "Ly6G", "TCRbeta", "CD19", "ASC", "Iba1", "EpCAM", "AF",
        ],
    "classes"  : [
        "Other", "Epithelium", "Neutrophil", "Macrophage", "Bcell", "Tcell",  
        ],
    
    "colors" : [
        [0.0, 0.0, 1.0], # Other
        [0.0, 0.8, 0.2], # Epithelium
        [1.0, 0.0, 0.0], # Neutrophil
        [1.0, 0.0, 1.0], # Macrophage
        [0.0, 1.0, 1.0], # Bcell
        [0.8, 0.8, 0.0], # Tcell
        ],
    
    # Paths
    "data_path" : Path("D:\local_mbl\data"),
    "model_preview_name" : "model-sm_preview_256_binary_227-1024",
    "model_nuclei_name"  : "model-sm_nuclei_256_edt_192-1024",
    "model_classes_name" : "model-cls_12_2_6_5045-8190b",
    
    # extract()
    "pvw_level" : 4,
    "stk_level" : 1,
        
    # predict_nuclei()
    "thresh_0"  : 0.25,
    "thresh_1"  : 0.50,    
    
    # Measure()
    "crop_size" : 32,

    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    def __init__(self, parameters=None):
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.extract()
        self.predict_nuclei()
        self.predict_classes()
        self.measure()
        self.analyse()
        
#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        qpt_paths = list(self.data_path.glob("*unmixed.qptiff"))

        self.qpt_paths = []
        self.mtd_paths = []
        self.vws_paths = []
        for p, qpt_path in enumerate(qpt_paths): 
            
            # Metadata --------------------------------------------------------
            
            # Paths
            mtd_name = str(qpt_path.name).replace(".qptiff", "_metadata.pkl")
            mtd_path = self.data_path / mtd_name
                        
            if not mtd_path.exists():
                mtd = get_slide_metadata(qpt_path)
                with open(mtd_path, "wb") as file:
                    pickle.dump(mtd, file)
            else:
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                    
            # Data ------------------------------------------------------------
                    
            for roi_idx in range(len(mtd["roi"])):
                
                # Paths
                vws_name = str(qpt_path.name).replace(".qptiff", f"_{roi_idx:02d}")
                vws_path = self.data_path / vws_name
                self.qpt_paths.append(qpt_path)
                self.mtd_paths.append(mtd_path)
                self.vws_paths.append(vws_path)
                
                pvw_raw_path = vws_path / "preview_raw.tif"
                pvw_msk_path = vws_path / "preview_mask.tif"
                pvw_nrm_path = vws_path / "preview_norm.tif"
                pvw_sts_path = vws_path / "preview_stats.pkl"
                stk_raw_path = vws_path / "stack_raw.npz"
                stk_nrm_path = vws_path / "stack_norm.npz"

                if not stk_nrm_path.exists():
                    
                    # Log
                    t0 = time.time()
                    log_str = f"Main() : extract() : {vws_path.name}"
                    print("-" * len(log_str))
                    print(log_str)
                    print("-" * len(log_str))
                    
                    # Model
                    if "model_preview" not in locals():
                        model_preview = Model(
                            parameters=None, 
                            model_path=Path.cwd() / self.model_preview_name
                            )
                    
                    vws_path.mkdir(parents=True, exist_ok=True) 
    
                    # Preview raw
                    pvw_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.pvw_level, roi_idx=roi_idx,
                        )
                    
                    # Preview mask
                    pvw_prp = prep_preview(pvw_raw, channels=[2, 5, 6])
                    pvw_msk = model_preview.predict(
                        pvw_prp, batch_size=32) > 0.5
                    pvw_msk = (pvw_msk * 255).astype("uint8")
                    
                    # Preview stats
                    pvw_sts = get_preview_stats(pvw_raw, pvw_msk, self.channels)
                    
                    # Preview normalization
                    pvw_nrm = (norm_data(pvw_raw, pvw_sts) * 255).astype("uint8")
                    
                    # Stack raw
                    stk_raw = load_view(
                        qpt_path, metadata=mtd, channels="all", 
                        level=self.stk_level, roi_idx=roi_idx,
                        )
                    stk_raw = norm_12_to_8bits(stk_raw)
                    
                    # Stack normalization
                    stk_nrm = (norm_data(stk_raw, pvw_sts) * 255).astype("uint8")
                    
                    # Save
                    io.imsave(pvw_raw_path, pvw_raw, check_contrast=False)
                    io.imsave(pvw_msk_path, pvw_msk, check_contrast=False)
                    io.imsave(pvw_nrm_path, pvw_nrm, check_contrast=False)
                    save_compressed_stack(stk_raw, stk_raw_path)
                    save_compressed_stack(stk_nrm, stk_nrm_path)
                    with open(pvw_sts_path, "wb") as file:
                        pickle.dump(pvw_sts, file)     
                        
                    t1 = time.time()
                    print(f"duration : {t1 - t0:.3f}s")
                            
#%% Class(Main) : predict_nuclei() --------------------------------------------
                   
    def predict_nuclei(self):
        
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            qpt_path = self.qpt_paths[p]
            mtd_path = self.mtd_paths[p]
            pvw_sts_path = vws_path / "preview_stats.pkl"
            ncl_msk_path = vws_path / "nuclei_mask.npz"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            roi_idx = int(vws_path.stem[-2:])
            
            if not ncl_ctd_path.exists():
                       
                # Log
                t0 = time.time()
                log_str = f"Main() : predict_nuclei() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))
                
                # Model
                if "model_nuclei" not in locals():
                    model_nuclei = Model(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_nuclei_name
                        )
                
                # Load
                with open(mtd_path, "rb") as file:
                    mtd = pickle.load(file)
                with open(pvw_sts_path, "rb") as file:
                    pvw_sts = pickle.load(file)
                ncl = load_view(
                    qpt_path, metadata=mtd, channels="DAPI", 
                    level=0, roi_idx=roi_idx,
                    )
                                   
                # Nuclei prediction
                ncl_prd = model_nuclei.predict(
                    norm_data(ncl, pvw_sts, channel="DAPI"), 
                    patch_overlap=32, batch_size=64, chunk_size=5000, 
                    )
    
                # Nuclei mask
                ncl_msk = get_nuclei_mask(
                    ncl_prd, thresh_0=self.thresh_0, thresh_1=self.thresh_1)
    
                # Nuclei centroids                    
                ncl_ctd = get_nuclei_centroids(
                    ncl_msk, n_levels=len(mtd["level"]))
                
                # Save
                save_compressed_mask(ncl_msk, ncl_msk_path)
                with open(ncl_ctd_path, "wb") as file:
                    pickle.dump(ncl_ctd, file)
    
                # Clear
                del ncl, ncl_prd, ncl_msk, ncl_ctd
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")
                                                         
#%% Class(Main) : predict_classes() -------------------------------------------
            
    def predict_classes(self):
        
        for p, vws_path in enumerate(self.vws_paths):
            
            # Paths
            stk_nrm_path = vws_path / "stack_norm.npz"
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
            ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
            
            if not ncl_ent_path.exists():
            
                # Log
                t0 = time.time()
                log_str = f"Main() : predict_classes() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))    
            
                # Model
                if "model_classes" not in locals():
                    model_classes = Model(
                        parameters=None, 
                        model_path=Path.cwd() / self.model_classes_name
                        )
                    
                # Load
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                stk_nrm = load_compressed_stack(stk_nrm_path)
                    
                # Get crops
                crop_size = int(self.model_classes_name.split("_")[1])
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=crop_size, 
                    stk_level=self.stk_level
                    )
                
                # Predict nuclei classes
                crp_nrm = prepare_crops(crp_nrm)
                ncl_prd = model_classes.predict(crp_nrm)
                ncl_ent = entropy(ncl_prd, axis=1, base=len(self.classes))
                ncl_prd = np.argmax(ncl_prd, axis=1)
    
                # Save
                with open(ncl_prd_path, "wb") as file:
                    pickle.dump(ncl_prd, file)
                with open(ncl_ent_path, "wb") as file:
                    pickle.dump(ncl_ent, file)
                    
                # Clear
                del stk_nrm, ncl_ctd, crp_nrm, ncl_prd, ncl_ent
                tf.keras.backend.clear_session()
                gc.collect()
                
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")

#%% Class(Main) : measure() ---------------------------------------------------

    def measure(self):
                
        for p, vws_path in enumerate(self.vws_paths):

            # Paths
            ncl_ctd_path = vws_path / "nuclei_centroids.pkl"
            ncl_sts_path = vws_path / "nuclei_stats.pkl"
            stk_nrm_path = vws_path / "stack_norm.npz"     

            if not ncl_sts_path.exists():           

                # Log
                t0 = time.time()
                log_str = f"Main() : measure() : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))                           
        
                # Load
                with open(ncl_ctd_path, "rb") as file:
                    ncl_ctd = pickle.load(file)
                stk_nrm = load_compressed_stack(stk_nrm_path)
                
                # Get crops
                crp_nrm = get_crops(
                    stk_nrm, ncl_ctd, 
                    crop_size=self.crop_size, 
                    crop_fraction=None, 
                    stk_level=self.stk_level,
                    )
                crp_nrm = crp_nrm.astype("float32") / 255 # normalize
                
                # Get nuclei stats
                ncl_sts = get_nuclei_stats(crp_nrm, self.channels)
                    
                # Save
                with open(ncl_sts_path, "wb") as file:
                    pickle.dump(ncl_sts, file)
                    
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")
                
#%% Class(Main) : analyse() ---------------------------------------------------  

    def analyse(self):
        
        # Paths
        mncl_df_path = self.data_path / "merged_nuclei_df.csv"
        mncl_dfa_path = self.data_path / "merged_nuclei_dfa.csv"
        mncl_dfn_path = self.data_path / "merged_nuclei_dfn.csv"
        mncl_dfan_path = self.data_path / "merged_nuclei_dfan.csv"
        
        if not mncl_dfan_path.exists(): 
        
            mncl_df, mncl_dfa, mncl_dfn, mncl_dfan = [], [], [], [],
            for p, vws_path in enumerate(self.vws_paths):
                
                # Paths
                ncl_prd_path = vws_path / "nuclei_classes_predict.pkl"
                ncl_ent_path = vws_path / "nuclei_classes_entropy.pkl"
                ncl_sts_path = vws_path / "nuclei_stats.pkl"
                ncl_df_path = vws_path / "nuclei_df.csv"
                ncl_dfa_path = vws_path / "nuclei_dfa.csv"
                ncl_dfn_path = vws_path / "nuclei_dfn.csv"
                ncl_dfan_path = vws_path / "nuclei_dfan.csv"
    
                # Log
                t0 = time.time()
                log_str = f"Main() : analyse : {vws_path.name}"
                print("-" * len(log_str))
                print(log_str)
                print("-" * len(log_str))   
                
                # Load
                with open(ncl_prd_path, "rb") as file:
                    ncl_prd = pickle.load(file)
                with open(ncl_ent_path, "rb") as file:
                    ncl_ent = pickle.load(file)
                with open(ncl_sts_path, "rb") as file:
                    ncl_sts = pickle.load(file)
                    
                # Format nuclei stats
                ncl_df, ncl_dfa, ncl_dfn, ncl_dfan = format_nuclei_stats(
                    ncl_prd, ncl_ent, ncl_sts, vws_path, 
                    self.channels, self.classes
                    )
                
                # # Plot
                # fig = plot(ncl_df_avg, ncl_df_avg_nrm, 
                #     self.channels, self.classes, self.colors)
                
                # Append
                mncl_df.append(ncl_df)
                mncl_dfa.append(ncl_dfa)
                mncl_dfn.append(ncl_dfn)
                mncl_dfan.append(ncl_dfan)
                                
                # Save
                ncl_df.to_csv(ncl_df_path, index=False)
                ncl_dfa.to_csv(ncl_dfa_path, index=False)
                ncl_dfn.to_csv(ncl_dfn_path, index=False)
                ncl_dfan.to_csv(ncl_dfan_path, index=False)
                # fig.savefig("nuclei_df_plot.png", dpi=300, bbox_inches="tight")
                
                t1 = time.time()
                print(f"duration : {t1 - t0:.3f}s")
                    
            # Merge dataframes
            mncl_df = pd.concat(mncl_df, axis=0, ignore_index=True)
            mncl_dfa = pd.concat(mncl_dfa, axis=0, ignore_index=True)
            mncl_dfn = pd.concat(mncl_dfn, axis=0, ignore_index=True)
            mncl_dfan = pd.concat(mncl_dfan, axis=0, ignore_index=True)
            
            # # # Plot
            # # fig = plot(mrg_ncl_df_avg, mrg_ncl_df_avg_nrm, 
            # #     self.channels, self.classes, self.colors)
            
            # Save
            mncl_df.to_csv(mncl_df_path, index=False)
            mncl_dfa.to_csv(mncl_dfa_path, index=False)
            mncl_dfn.to_csv(mncl_dfn_path, index=False)
            mncl_dfan.to_csv(mncl_dfan_path, index=False)
            # # fig.savefig("merged_nuclei_df_plot.png", dpi=300, bbox_inches="tight")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(parameters=main_parameters)
    
#%% Load ----------------------------------------------------------------------
    
    vws_idx = 8

    # Paths
    if isinstance(vws_idx, int):
        path = main.vws_paths[vws_idx]
        prefix = ""
    elif vws_idx == "all":
        path = main.data_path
        prefix = "merged_"

    # Load
    df = pd.read_csv(path / f"{prefix}nuclei_df.csv")
    dfn = pd.read_csv(path / f"{prefix}nuclei_dfn.csv")
    dfa = pd.read_csv(path / f"{prefix}nuclei_dfa.csv")
    dfan = pd.read_csv(path / f"{prefix}nuclei_dfan.csv")   
    
#%% Plot function(s) ----------------------------------------------------------
    
    import matplotlib.pyplot as plt

    def plot(df, dfa, dfn, dfan, channels, classes, colors):
        
        # Inputs ------------------------------------------------------------------
        
        gcolor = [0.5, 0.5, 0.5]
        schannels = ["Ly6G", "TCRbeta", "CD19", "Iba1", "EpCAM"]
        sclasses = ["Other", "Epi.", "Neut.", "Macro.", "Bcell", "Tcell"]
        merged = df["view"].nunique(dropna=False) > 1
        
        # Nested function(s) ------------------------------------------------------
        
        def _plot(ax, vals, idx, color, scatter=False, merged=False):
            
            if scatter:
            
                # Scatter
                xs = np.random.normal(loc=idx, scale=0.05, size=len(vals))
                ax.scatter(
                    xs, vals, 
                    facecolors=color, edgecolors="none", 
                    s=10, alpha=1.0, zorder=0
                    )
            
            # Boxplot
            if merged:
                medianprops = dict(color="k", linewidth=1.0)
            else:
                medianprops = dict(color=color, linewidth=2.0)
            ax.boxplot(
                vals, positions=[idx], widths=0.5, patch_artist=True,
                boxprops=dict(color="0.5", facecolor="none", linewidth=1.0),
                medianprops=medianprops,
                whiskerprops=dict(color=gcolor, linewidth=1.0),
                capprops=dict(color=gcolor, linewidth=1.0),
                showfliers=False
                )
            
            # Mean line
            ax.axhline(y=1, color=gcolor, linestyle=":", linewidth=1)
            
            # x grid
            ax.grid(axis="x", linestyle=":", alpha=0.6, color=gcolor)
        
        # Execute -----------------------------------------------------------------
        
        # Initialize plot
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 10))
        axes = axes.flatten()
        
        for cl, cla in enumerate(classes):
    
            ax = axes[cl]                   
            dfs = dfan if merged else dfn
            
            for ch, chn in enumerate(schannels):
                vals = np.array(dfs[dfs["class"] == cla][f"{chn}_avg"])
                _plot(ax, vals, ch, colors[cl], scatter=False, merged=merged)
                        
            # Legend
            if merged:
                ax.set_ylim(0, 6)
            ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
            ax.set_xticklabels(schannels, rotation=45, ha="center")
            ax.set_title(cla, pad=10)
        
        for i, measure in enumerate(["count", "entropy", "ASC_avg"]):
            
            ax = axes[i + 6] 
            
            if measure == "count":
                dfs = dfan
            if measure == "entropy":
                dfs = dfa if merged else df
            if measure == "ASC_avg":
                dfs = dfan if merged else dfn
            
            for cl, cla in enumerate(classes):
                vals = dfs[dfs["class"] == cla][measure]
                _plot(ax, vals, cl, colors[cl], merged)
                
            # Legend
            if measure == "count":
                ax.set_ylim(0, 1)
                ax.set_ylabel("Count")
                ax.set_title("Relative count", pad=10)
            if measure == "entropy":
                ax.set_ylabel("Entropy")
                ax.set_title("Class pred. entropy", pad=10)
            if measure == "ASC_avg":
                if merged:
                    ax.set_ylim(0, 6)
                ax.set_ylabel("Mean Norm. Fluo. Int. (A.U.)")
                ax.set_title("ASC Fluo. Int.", pad=10)
            ax.set_xticklabels(sclasses, rotation=45, ha="center")
    
        # Main title    
        fig.suptitle(
            "Merged views" if merged else dfa["view"].iloc[0],
            fontsize=16, fontweight="bold", color="k",
            y=0.95, x=0.025, ha="left",
            )
    
        # Secondary text
        text = (
            f"n view(s) = {dfa['view'].nunique()}\n"
            f"n nuclei = {dfa['count'].sum():,}\n"
            f"avg. entropy = {dfa['entropy'].mean():.3f}"
            )
        fig.text(
            0.05, 0.92, text,
            fontsize=12, style="italic", color=gcolor,
            ha="left", va="top",
            )
    
        plt.tight_layout(rect=[0, 0, 1, 0.9])
    
        return fig

#%% Plot ----------------------------------------------------------------------
    
    plot(df, dfa, dfn, dfan, main.channels, main.classes, main.colors)
