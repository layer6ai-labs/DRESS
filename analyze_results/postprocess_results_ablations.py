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

    latex_table = "\\toprule \n"
    # full ablation table
    latex_table += "Method & Shapes3D & Causal3D & MPI3D-Hard & CelebA-Hair & CelebA-Primary \\\\ \n\midrule \n"

    for method, res_dict in ACCURACIES_ALL.items():
        if method == 'DRESS':
            latex_table += "DRESS & "
        elif method == 'Ablate-Disentangle':
            latex_table += "\\makecell[l]{DRESS w/o \\\\ \hspace{4pt} Disent. Repsent.} & "
        elif method == 'Ablate-Align':
            latex_table += "\\makecell[l]{DRESS w/o \\\\ \hspace{4pt} Lat. Dim. Align.} & "
        elif method == 'Ablate-Individual-Cluster':
            latex_table += "\\makecell[l]{DRESS w/o \\\\ \hspace{4pt} Ind. Dim. Cluster.} & "
        else:
            continue
        # full ablation table
        for ds in ['shapes3d', 'causal3d', 'mpi3dhard', 'celebahair', 'celebaprimary']:
            if method == "Ablate-Align" and ds in ['shapes3d', 'causal3d', 'mpi3dhard']:
                # this ablation study doesn't apply
                latex_table += "-"
            else:
                res_vals = res_dict["five-shot"][ds]
                if len(res_vals) == 0:
                    latex_table += "TODO"
                else:
                    res_avg = np.mean(res_vals)
                    res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                    if method == "DRESS":
                        # bold font
                        latex_table += f"\\textbf{{{res_avg:.1f}}}\% {{\scriptsize $\pm$ \\textbf{{{res_std:.1f}}}}}\%"
                    else:
                        latex_table += f"${res_avg:.1f}\%$ {{\scriptsize $\pm {res_std:.1f}\%$}}"
            # full ablation table
            if ds != "celebaprimary":
                latex_table += " & "
            else:
                latex_table += "\\\\ \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n\n\n" 
    
    # write the results to a file 
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)

    
    print(f"Latex table generated and saved to {latex_table_filename} file!")
    print("Script finished!")