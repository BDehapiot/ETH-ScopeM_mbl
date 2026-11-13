#%% Imports -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#%% Class(Plot) ---------------------------------------------------------------

class Plot:
    def __init__(
            self, main, 
            ncl_data_mrg, ncl_data_avg, ncl_data_std, ncl_data_sem,
            asc_data_mrg, asc_data_avg, asc_data_std, asc_data_sem,
            ncl_dst_bin, asc_sts_mrg,
            asc_sts_avg=None,
            asc_sts_std=None,
            asc_sts_sem=None,
            merged=False,
            ):
        
        self.main = main
        self.parameters = main.parameters
        for key, val in self.parameters.items():
            setattr(self, key, val)
            
        self.ncl_data_mrg = ncl_data_mrg
        self.ncl_data_avg = ncl_data_avg
        self.ncl_data_std = ncl_data_std
        self.ncl_data_sem = ncl_data_sem
        self.asc_data_mrg = asc_data_mrg
        self.asc_data_avg = asc_data_avg
        self.asc_data_std = asc_data_std
        self.asc_data_sem = asc_data_sem
        self.ncl_dst_bin  = ncl_dst_bin
        self.asc_sts_mrg  = asc_sts_mrg
        self.asc_sts_avg  = asc_sts_avg
        self.asc_sts_std  = asc_sts_std
        self.asc_sts_sem  = asc_sts_sem
        self.merged = merged
        self.fig = None
            
        # Run
        self.run()
                
#%% Class(Plot) : plot function(s) --------------------------------------------
                
    def barplot(
            self, x, y, ax=0, err=None,
            color="gray", width=0.5, shift=None, 
            ):
        
        if shift is not None:
            x = np.arange(len(x)) + shift

        ax.bar(x, y, color=color, width=width, zorder=2)
        
        if not err is None:
            ax.errorbar(
                x, y, yerr=err, 
                capsize=3, elinewidth=1, ecolor="0.0", 
                fmt="none", zorder=1,
                )
            
        ax.set_axisbelow(True)
        ax.axhline(**self.hline_params)
        ax.grid(**self.grid_params)
            
    def lineplot(
            self, x, y, ax=0, 
            color="gray", label="label",
            ):
        
        ax.plot(x, y, color=color, label=label)
        ax.axhline(**self.hline_params)
        ax.grid(**self.grid_params)
    
    def boxplot(
            self, x, y, ax=0, 
            color="gray", add_dots=False,             
            ):
        
        linecolor = "0.5"
        mediancolor = "0.5" if add_dots else color
        medianwidth = 1 if add_dots else 2
        
        ax.boxplot(
            y, positions=[x], widths=0.5, patch_artist=True,
            boxprops=dict(color=linecolor, facecolor="none", linewidth=1),
            medianprops=dict(color=mediancolor, linewidth=medianwidth),
            whiskerprops=dict(color=linecolor, linewidth=1),
            capprops=dict(color=linecolor, linewidth=1),
            showfliers=False,
            )
        
        # Add dots
        if add_dots:
            x_jittered = x + np.random.normal(0, 0.08, size=len(y))
            ax.scatter(
                x_jittered, y,
                color=color, alpha=0.75, s=10, edgecolors="none",
                )
        
        ax.axhline(**self.hline_params)
        ax.grid(**self.grid_params)   
        
#%% Class(Plot) : run() -------------------------------------------------------
                
    def run(self):
        
        plt.ioff()
        
        # Init.
        self.fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 12))
        axes = axes.flatten()
        
        # Main title/info -----------------------------------------------------
           
        # Main title    
        self.fig.suptitle(
            "Merged views" if self.merged else self.ncl_data_mrg["view"].iloc[0],
            fontsize=16, fontweight="bold", color="k",
            y=0.92, x=0.025, ha="left",
            )

        # Secondary text
        if self.merged:    
            ncl_n = self.ncl_data_mrg["count"].sum() 
        else:
            ncl_n = len(self.ncl_data_mrg)
        nsp_n = self.asc_data_avg.loc[
            self.asc_data_avg["class"] == "Non-speckle", "count"
            ].values[0]
        spk_n = self.asc_data_avg.loc[
            self.asc_data_avg["class"] == "Speckle", "count"
            ].values[0]
        text = (
            f"views = {self.ncl_data_mrg['view'].nunique()} | "
            f"nuclei = {ncl_n:,} | "
            f"non-speckles = {nsp_n:,} | "
            f"speckles = {spk_n:,}"
            )
        self.fig.text(
            0.025, 0.89, text,
            fontsize=12, color="0.25",
            ha="left", va="top",
            )
        
#%% Class(Plot) : (0) Class predictions frequency -----------------------------
        
        # Data
        ax = axes[0]
        x = self.plot_classes
        y = self.ncl_data_avg["freq"] 
        if self.err_type == "std":
            err = self.ncl_data_std["freq"] if self.merged else None
        if self.err_type == "sem":
            err = self.ncl_data_sem["freq"] if self.merged else None
        
        # Plot
        self.barplot(
            x, y, ax=ax, err=err,
            color=self.ncl_colors, width=0.5, shift=None
            )
        
        # Legend
        ax.set_ylim(0, 1)
        ax.set_ylabel("Frequency")
        ax.set_xticks(np.arange(0, len(self.plot_classes)))
        ax.set_xticklabels(self.plot_classes, rotation=45, ha="center")
        ax.set_title("Ncl. class pred. freq.", pad=10)
        
#%% Class(Plot) : (1) Class predictions entropy -------------------------------
        
        # Data
        ax = axes[1]
        x = self.plot_classes
        y = self.ncl_data_avg["entropy"] 
        if self.err_type == "std":
            err = self.ncl_data_std["entropy"]
        if self.err_type == "sem": 
            err = self.ncl_data_sem["entropy"]
        
        # Plot
        self.barplot(
            x, y, ax=ax, err=err,
            color=self.ncl_colors, width=0.5, shift=None
            )
                    
        # Legend
        ax.set_ylim(0, 1)
        ax.set_ylabel("Entropy")
        ax.set_xticks(np.arange(0, len(self.plot_classes)))
        ax.set_xticklabels(self.plot_classes, rotation=45, ha="center")
        ax.set_title("Ncl. class pred. entropy", pad=10)
        
#%% Class(Plot) : (2) ASC statistics ------------------------------------------
        
        # Inputs
        ymax = 0.10
        
        # Data
        ax = axes[2]
        x = ["non-spk", "spk"] * 2
        tags = ["nsp_area", "spk_area", "nsp_num", "spk_num"]
        
        for t, tag in enumerate(tags):
            
            key = f"asc_{tag}_ratio"
            
            # Data 
            if self.merged:
                y = self.asc_sts_avg[key]
                if self.err_type == "std":
                    err = self.asc_sts_std[key]
                if self.err_type == "sem": 
                    err = self.asc_sts_sem[key]
            else:
                y = self.asc_sts_mrg[key]
                err = None
                
            # Plot
            self.barplot(
                t, y, ax=ax, err=err,
                color="0.5", width=0.5, shift=None
                )
            
            # Text
            if t in [1, 3]:
                if t == 1: label = "asc/tiss. \narea ratio"
                if t == 3: label = "acs/nucl. \nnum. ratio"
                ax.text(
                    t - 0.5, ymax * 0.95, label,
                    fontsize=10, color="black",
                    horizontalalignment="center",
                    verticalalignment="top",
                    )
            
        # Legend
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Density")
        ax.set_xticks(np.arange(0, len(tags)))
        ax.set_xticklabels(x, rotation=45, ha="center")
        ax.set_title("ASC density", pad=10)
        
#%% Class(Plot) : (3) Class ASC intensities -----------------------------------
        
        # Inputs
        ymax = 15
        norm_str = "(norm.)" if self.int_norm else ""
        
        # Data
        ax = axes[3]
        x = self.plot_classes
        
        for t, tag in enumerate(["nsp", "spk"]):
            
            key = f"asc_{tag}_int"
            shift = -0.2 if t == 0 else 0.2
            
            # Data
            y = self.ncl_data_avg[key]
            if self.err_type == "std":
                err = self.ncl_data_std[key]
            if self.err_type == "sem": 
                err = self.ncl_data_sem[key]
            
            self.barplot(
                x, y, ax=ax, err=err,
                color=self.ncl_colors, width=0.3, shift=shift
                )
                
            # Text
            ax.text(
                0.5, 0.95, "non-speckle vs. speckle",
                fontsize=10, color="black",
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="top",
                )
            
        # Legend
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Fluo. Int. (A.U.)")
        ax.set_xticks(np.arange(0, len(self.plot_classes)))
        ax.set_xticklabels(self.plot_classes, rotation=45, ha="center")
        ax.set_title(f"Ncl. ASC int. {norm_str}", pad=10)
            
#%% Class(Plot) : (4-5) ASC distance class enrichment -------------------------
        
        for i in [4, 5]:
        
            # Inputs
            if i == 4:
                ymax = 5
                short_tag, long_tag = "nsp", "non-spk."
            if i == 5:
                ymax = 2.5
                short_tag, long_tag = "spk", "spk."
            norm_str = "(norm.)" if self.dst_norm else ""
                
            # Data
            ax = axes[i]
            x = self.ncl_dst_bin["bins"]  
            
            for cl, cla in enumerate(self.ncl_classes):
                y = self.ncl_dst_bin[cla + f"_asc_{short_tag}_dst"]
                self.lineplot(
                    x, y, ax=ax, 
                    color=self.ncl_colors[cl], 
                    label=self.plot_classes[cl],
                    )
                
            # Legend
            ax.set_ylim(0, ymax)
            ax.set_ylabel("Fold change")
            ax.set_xlabel("distance (µm)")
            ax.set_xticks(x)
            ax.set_title(f"{long_tag} dist. to ncl. {norm_str}", pad=10)
            ax.legend(
                loc="upper right", 
                frameon=False, 
                fontsize=8, 
                ncol=2,
                columnspacing=1.0,
                )
            
#%% Class(Plot) : (6-11) Channel intensities ----------------------------------
        
        def get_ylim(ax, min_ymax=6):
            _, ymax = ax.get_ylim()
            ax.set_ylim(0, np.maximum(min_ymax, ymax))  
        
        for i, ii in enumerate([6, 7, 8, 9, 10, 11]): 
        
            # Inputs    
            cla = self.ncl_classes[i] if self.merged else i
            add_dots = True if self.merged else False
            norm_str = "(norm.)" if self.int_norm else ""
        
            # Data
            ax = axes[ii]
            x = self.plot_classes
            
            for ch, chn in enumerate(self.plot_channels):
                y = np.array(self.ncl_data_mrg[
                    self.ncl_data_mrg["class"] == cla][f"{chn}_int"])
                self.boxplot(
                    ch, y, ax=ax, 
                    color=self.ncl_colors[i], add_dots=add_dots,
                    )
        
            # Legend
            get_ylim(ax)
            ax.set_ylabel("Fluo. Int. (A.U.)")
            ax.set_xticklabels(self.plot_channels, rotation=45, ha="center")
            ax.set_title(self.ncl_classes[i] + " " + norm_str, pad=10)
        
#%% Terminate -----------------------------------------------------------------

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.ion()