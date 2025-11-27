import numpy as np
import os
import matplotlib.pyplot as plt

from res_all import ACCURACIES_ALL, DIVERSITIES_ALL
from compute_performance_diversity_corr import compute_performance_metric_correlation

methods_to_include = ['DRESS', 'Supervised-All', "CACTUS-DC", "CACTUS-DINO"]
methods_markers = ['o', 's', '^', 'P']
assert len(methods_to_include) == len(methods_markers)
datasets_to_include = ['smallnorb', 'shapes3d', 'causal3d', 'mpi3dhard', 'celebahair']
datasets_to_display = ['SmallNORB', 'Shapes3D', 'Causal3D', 'MPI3D-Hard', 'CelebA-Hair']

if __name__ == "__main__":

    """
    Plot correlations between task diversities and few-shot adaptation performances
    """
    n_rows = 1
    n_cols = int(len(datasets_to_include) / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5*n_cols, 5*n_rows))
    for ds_idx, ds in enumerate(datasets_to_include):
        if n_rows == 1:
            ax = axes[ds_idx]
        else:
            row_idx, col_idx = ds_idx // n_cols, ds_idx % n_cols
            ax = axes[row_idx, col_idx]
        # set the axis title and grid
        ax.set_title(f"Dataset: {datasets_to_display[ds_idx]}", fontsize=22)
        ax.grid(True, alpha=0.3)

        diversity_vals_all_methods, accur_vals_all_methods = [], []
        for method in methods_to_include:
            # for adaptation performance sharing the meta-training set
            if ds == "mpi3deasy":
                diversity_vals_all_methods.append(1-np.mean(DIVERSITIES_ALL[method]['mpi3dhard']))
            elif ds in ["celebaprimary", "lfwacrossdomain"]:
                diversity_vals_all_methods.append(1-np.mean(DIVERSITIES_ALL[method]['celebahair']))
            else:
                diversity_vals_all_methods.append(1-np.mean(DIVERSITIES_ALL[method][ds]))
            # take average of the performance across different shots
            accurs_vals_all_shots = []
            for shot in ['five-shot', 'ten-shot']:
                accurs_vals_all_shots.append(np.mean(ACCURACIES_ALL[method][shot][ds]))
            accur_vals_all_methods.append(np.mean(accurs_vals_all_shots))
        # plot the scatter points
        for x, y, marker in zip(diversity_vals_all_methods, 
                                accur_vals_all_methods, 
                                methods_markers):
            ax.scatter(x, y, label=method, marker=marker, s=90)
        # compute and show the correlation
        correlation = compute_performance_metric_correlation(
                            performances=accur_vals_all_methods, 
                            metrics=diversity_vals_all_methods
                        )
        ax.text(0.05, 0.87, f"Corr: {correlation:.2f}", 
                transform=ax.transAxes, fontsize=23, verticalalignment='top')
        # fit the line
        slope, intercept = np.polyfit(diversity_vals_all_methods, accur_vals_all_methods, 1)
        x_line = np.array([min(diversity_vals_all_methods), max(diversity_vals_all_methods)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linestyle='--', color='gray')
        
    # shared labels for x and y axes
    if n_rows == 1:
        for ax in axes:
            ax.set_xlabel("Diversity Score", fontsize=20)
        axes[0].set_ylabel("Few-Shot Accuracy (%)", fontsize=20)
    else:
        for ax in axes[n_rows-1, :]:
            ax.set_xlabel("Diversity Score", fontsize=20)
        for ax in axes[:, 0]:
            ax.set_ylabel("Few-Shot Accuracy (%)", fontsize=20)
    # create a single legend for the methods on the top right subplot of the figure
    if n_rows == 1:
        axes[0].legend(methods_to_include, fontsize=17, loc='lower right')
    else:
        axes[0, 0].legend(methods_to_include, fontsize=17, loc='lower right')
    # increase the axis tick font size
    for ax in axes.flat: 
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    plt.tight_layout()
    # save the figure 
    figure_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "res_plot_correlation.pdf")
    plt.savefig(figure_filename)
    print(f"Figure saved to {figure_filename} file!")
            
    print("Script finished!")