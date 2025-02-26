# Script to organize few-shot learning results of all the methods
# to a latex table file

import os
import sys
sys.path.append("../")
from utils import *

results = {
    'Supervised-Original': {
        'shapes3d': [0.05],
        'mpi3dhard': [0.07],
        'smallnorb': [0.00],
        'causal3d': [0.02],
        'celebahair': [0.12]
    },
    'Supervised-All': {
        'shapes3d': [0.01],
        'mpi3dhard': [0.01],
        'smallnorb': [0.02],
        'causal3d': [0.01],
        'celebahair': [0.16]
    },
    'Supervised-Oracle': {
        'shapes3d': [0.01],
        'mpi3dhard': [0.03],
        'smallnorb': [0.00],
        'causal3d': [0.02],
        'celebahair': [0.17]
    },
    'CACTUS-DeepCluster': {
        'shapes3d': [0.20],
        'mpi3dhard': [0.21],
        'smallnorb': [0.32],
        'causal3d': [0.12],
        'celebahair': [0.08]
    },
    'CACTUS-DinoV2': {
        'shapes3d': [0.39],
        'mpi3dhard': [0.42],
        'smallnorb': [0.42],
        'causal3d': [0.37],
        'celebahair': [0.26]
    },
    'DRESS': {
        'shapes3d': [0.10],
        'mpi3dhard': [0.08],
        'smallnorb': [0.30],
        'causal3d': [0.27],
        'celebahair': [0.02]
    }
}


if __name__ == "__main__":
    print(f"Processing results...")

    latex_table = "\\toprule \n"
    latex_table += "Method & Shapes3D & MPI3D-Hard & SmallNORB & Causal3D & CelebA-Hair \\\\ \n\midrule \n"

    for method, res_dict in results.items():
        if method == 'DRESS':
            latex_table += "\\textbf{DRESS} & "
        else:
            latex_table += f"{method} & "
        for ds in ['shapes3d', 'mpi3dhard', 'smallnorb', 'causal3d', 'celebahair']:
            res_vals = res_dict[ds]
            if len(res_vals) == 0:
                latex_table += "TODO"
            else:
                res_avg = np.mean(res_vals)
                res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                if (method == "DRESS" and ds in ['shapes3d', 'mpi3dhard', 'smallnorb', 'celebahair']) or \
                    (method == "CACTUS-DeepCluster" and ds == 'causal3d'):
                    # bold font
                    latex_table += f"\\textbf{{{1-res_avg:.2f}}} $\pm$ \\textbf{{{res_std:.2f}}}"
                else:
                    latex_table += f"${1-res_avg:.2f} \pm {res_std:.2f}$"
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