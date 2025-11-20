# Script to organize few-shot learning results of all the methods
# to a latex table file

import os
import sys
sys.path.append("../")
from utils import *
from res_all import DIVERSITIES_ALL


if __name__ == "__main__":
    print(f"Processing results...")

    latex_table = "\\toprule \n"
    latex_table += "Method & SmallNORB & Shapes3D & Causal3D & MPI3D-Hard & CelebA-Hair \\\\ \n\midrule \n"

    for method, res_dict in DIVERSITIES_ALL.items():
        if method == 'DRESS':
            latex_table += "\\textbf{DRESS} & "
        else:
            latex_table += f"{method} & "
        for ds in ['smallnorb', 'shapes3d', 'causal3d', 'mpi3dhard', 'celebahair']:
            res_vals = res_dict['within-metatrain'][ds]
            if len(res_vals) == 0:
                latex_table += "TODO"
            else:
                res_avg = np.mean(res_vals)
                res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                if (method == "DRESS" and ds in ['shapes3d', 'mpi3dhard', 'smallnorb', 'celebahair']) or \
                    (method == "CACTUS-DC" and ds == 'causal3d'):
                    # bold font
                    latex_table += f"\\textbf{{{1-res_avg:.2f}}} {{\scriptsize $\kern0.13em\pm$  \\textbf{{{res_std:.2f}}}}}"
                else:
                    latex_table += f"${1-res_avg:.2f}$ {{\scriptsize $\pm {res_std:.2f}$}}"
            if ds != "celebahair":
                latex_table += " & "
            else:
                latex_table += "\\\\ \n"
        if method == "Supervised-Oracle":
            latex_table += "\hline \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n" 
    
    # write the results to a file 
    latex_table_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "res_table_diversity.tex")
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")