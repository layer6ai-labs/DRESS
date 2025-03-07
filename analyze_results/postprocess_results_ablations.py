# Script to organize few-shot learning results of all the methods
# to a latex table file

import os
import sys
sys.path.append("../")
from utils import *
from res_all import ACCURACIES_ALL



if __name__ == "__main__":
    print(f"Processing results...")
    latex_table_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "res_table_ablations.tex")

    ################## Ablation I: Disentangled representations ##################
    latex_table = "\\toprule \n"
    latex_table += "Method & Causal3D \\\\ \n\midrule \n"

    for method, res_dict in ACCURACIES_ALL.items():
        if method == 'DRESS':
            latex_table += "DRESS & "
        elif method == 'Ablate-Disentangle':
            latex_table += "DRESS w/o Dis. Rep. & "
        else:
            continue
        res_vals = res_dict['causal3d']
        if len(res_vals) == 0:
            latex_table += "TODO"
        else:
            res_avg = np.mean(res_vals)
            res_std = np.std(res_vals) / np.sqrt(len(res_vals))
            if method == "DRESS":
                # bold font
                latex_table += f"\\textbf{{{res_avg:.2f}}}\% $\pm$ \\textbf{{{res_std:.2f}}}\%"
            else:
                latex_table += f"${res_avg:.2f}\% \pm {res_std:.2f}\%$"
        latex_table += "\\\\ \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n\n\n" 
    

    ################## Ablation II: Latent Alignments ##############
    latex_table += "\\toprule \n"
    latex_table += "Method & CelebA-Hair \\\\ \n\midrule \n"

    for method, res_dict in ACCURACIES_ALL.items():
        if method == 'DRESS':
            latex_table += "DRESS & "
        elif method == 'Ablate-Align':
            latex_table += "DRESS w/o LDA & "
        else:
            continue
        res_vals = res_dict['celebahair']
        if len(res_vals) == 0:
            latex_table += "TODO"
        else:
            res_avg = np.mean(res_vals)
            res_std = np.std(res_vals) / np.sqrt(len(res_vals))
            if method == "DRESS":
                # bold font
                latex_table += f"\\textbf{{{res_avg:.2f}}}\% $\pm$ \\textbf{{{res_std:.2f}}}\%"
            else:
                latex_table += f"${res_avg:.2f}\% \pm {res_std:.2f}\%$"
        latex_table += "\\\\ \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n\n\n" 
    

    ############### Ablation III: Individual Dimension Clustering ##########
    latex_table += "\\toprule \n"
    latex_table += "Method & CelebA-Primary \\\\ \n\midrule \n"

    for method, res_dict in ACCURACIES_ALL.items():
        if method == 'DRESS':
            latex_table += "DRESS & "
        elif method == 'Ablate-Individual-Cluster':
            latex_table += "DRESS w/o Ind. Dim. Cluster. & "
        else:
            continue
        res_vals = res_dict['celebaprimary']
        if len(res_vals) == 0:
            latex_table += "TODO"
        else:
            res_avg = np.mean(res_vals)
            res_std = np.std(res_vals) / np.sqrt(len(res_vals))
            if method == "DRESS":
                # bold font
                latex_table += f"\\textbf{{{res_avg:.2f}}}\% $\pm$ \\textbf{{{res_std:.2f}}}\%"
            else:
                latex_table += f"${res_avg:.2f}\% \pm {res_std:.2f}\%$"
        latex_table += "\\\\ \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n" 
    
    # write the results to a file 
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)

    
    print(f"Latex table generated and saved to {latex_table_filename} file!")
    print("Script finished!")