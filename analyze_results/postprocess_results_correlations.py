import numpy as np
import os
import matplotlib.pyplot as plt

from res_all import ACCURACIES_ALL, DIVERSITIES_ALL
from compute_performance_diversity_corr import compute_performance_diversity_correlation

plot_figure = True
write_table = False
methods_to_include = ['DRESS', 'Supervised-All', "CACTUS-DC", "CACTUS-DINO"]
methods_markers = ['o', 's', '^', 'D']
datasets_to_include = ['smallnorb', 'shapes3d', 'causal3d', 'mpi3dhard', 'celebahair']
datasets_to_display = ['SmallNORB', 'Shapes3D', 'Causal3D', 'MPI3D-Hard', 'CelebA-Hair']

if __name__ == "__main__":
    print(f"Processing results...")

    if write_table:
        latex_table = "\\toprule \n"
        latex_table += "Setup & SmallNORB & Shapes3D & Causal3D & MPI3D-Hard & CelebA-Hair \\\\ \n\midrule \n"
        for shot in ['five-shot', "ten-shot"]:
            if shot == 'five-shot':
                latex_table += "5-Shot & "
            else:   
                latex_table += "10-Shot & "
            for ds in datasets_to_include:
                accur_vals_all_methods, diversity_vals_all_methods = [], []
                for method in methods_to_include:
                    accur_val, diversity_val = ACCURACIES_ALL[method][shot][ds], \
                                                    DIVERSITIES_ALL[method][ds]
                    accur_vals_all_methods.append(accur_val)
                    diversity_vals_all_methods.append(diversity_val)
                # compute correlations over different seed trials
                res_vals = []
                for accur_vals_list, diversity_vals_list in zip(
                        zip(*accur_vals_all_methods), 
                        zip(*diversity_vals_all_methods)):
                    res_vals.append(
                        compute_performance_diversity_correlation(
                            np.array(accur_vals_list), 
                            1-np.array(diversity_vals_list))
                    )
                if len(res_vals) == 0:
                    latex_table += "TODO"
                else:
                    res_avg = np.mean(res_vals)
                    res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                    latex_table += f"${res_avg:.2f}$ {{\\scriptsize $\pm {res_std:.2f}$}}"
                if ds != "celebahair":
                    latex_table += " & "
                else:
                    latex_table += "\\\\ \n"

        # finishing the table by an underline
        latex_table += "\\bottomrule \n"

        # write the results to a file 
        latex_table_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "res_table_correlations.tex")
        with open(latex_table_filename, "w") as f:
            f.write(latex_table)
        
        print(f"Latex table generated and saved to {latex_table_filename} file!")
    if plot_figure:
        fig, axes = plt.subplots(2, len(datasets_to_include), 
                                 figsize=(5*len(datasets_to_include), 5*2),
                                 )
        for ds_idx, ds in enumerate(datasets_to_include):
            diversity_vals_all_methods = []
            for method in methods_to_include:
                diversity_vals_all_methods.append(1-np.mean(DIVERSITIES_ALL[method][ds]))
            for i, shot in enumerate(['five-shot', "ten-shot"]):
                ax = axes[i, ds_idx]
                accur_vals_all_methods = []
                for method in methods_to_include:
                    accur_vals_all_methods.append(np.mean(ACCURACIES_ALL[method][shot][ds]))
                # plot the scatter points
                for x, y, marker in zip(diversity_vals_all_methods, 
                                        accur_vals_all_methods, 
                                        methods_markers):
                    ax.scatter(x, y, label=method, marker=marker, s=100)
                # compute and show the correlation
                correlation = compute_performance_diversity_correlation(
                                    np.array(accur_vals_all_methods), 
                                    np.array(diversity_vals_all_methods))
                ax.text(0.05, 0.85, f"Corr: {correlation:.2f}", 
                        transform=ax.transAxes, fontsize=18, verticalalignment='top')
                # fit the line on each shot separately
                slope, intercept = np.polyfit(diversity_vals_all_methods, accur_vals_all_methods, 1)
                x_line = np.array([min(diversity_vals_all_methods), max(diversity_vals_all_methods)])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, linestyle='--', color='gray')
                ax.grid(True, alpha=0.3)
            # set the axis title and grid
            axes[0, ds_idx].set_title(f"Dataset: {datasets_to_display[ds_idx]}", fontsize=16)
        # shared labels for x and y axes
        for ax in axes[1, :]:
            ax.set_xlabel("Diversity Score", fontsize=14)
        for ax in axes[:, 0]:
            ax.set_ylabel("Few-Shot Accuracy (%)", fontsize=14)
        # create a single legend for the methods on the top right subplot of the figure
        axes[0, -1].legend(methods_to_include, fontsize=12, loc='lower right')

        fig.suptitle("Few-Shot Performance vs. Diversity Correlations", fontsize=20)
        plt.tight_layout()
        # save the figure 
        figure_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "res_plot_correlation.pdf")
        plt.savefig(figure_filename)
        print(f"Figure saved to {figure_filename} file!")
            
    print("Script finished!")