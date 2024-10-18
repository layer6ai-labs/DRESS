# Script to organize few-shot learning results of all the methods
# to a latex table file

import matplotlib.pyplot as plt
import os
from utils import *

results = {
    'Supervised-Original': {
        'shapes3d': [66.39, 62.60, 57.70, 61.44],
        'mpi3deasy': [56.78, 57.22, 57.79, 59.23],
        'mpi3dhard': [62.11, 66.96, 63.85, 60.15],
        'celebahard': [64.46]
    },
    'Supervised-All': {
        'shapes3d': [99.91, 99.86, 99.96, 99.98],
        'mpi3deasy': [99.86, 99.61, 99.35, 98.34],
        'mpi3dhard': [88.37, 87.38, 95.94, 92.41],
        'celebahard': [78.20]
    },
    'Supervised-Oracle': {
        'shapes3d': [99.99, 99.96, 100.00, 99.92],
        'mpi3deasy': [100.00, 100.00, 100.00, 100.00],
        'mpi3dhard': [99.32, 99.13, 99.72, 99.50],
        'celebahard': [87.16]
    },
    'Few-Shot Direct Adaptation': {
        'shapes3d': [63.43, 63.30, 72.79, 63.28],
        'mpi3deasy': [61.19, 60.63, 59.63, 60.90],
        'mpi3dhard': [62.57, 62.45, 61.30, 62.74],
        'celebahard': [64.69]
    },
    'Pre-Training and Fine-Tuning': {
        'shapes3d': [65.35, 56.26, 55.77, 54.15],
        'mpi3deasy': [91.30, 93.13, 93.73, 93.54],
        'mpi3dhard': [81.98, 79.52, 78.40, 78.10],
        'celebahard': []
    },
    'CACTUS-DeepCluster': {
        'shapes3d': [88.03, 86.73, 84.62, 87.86],
        'mpi3deasy': [84.18, 86.27, 85.80, 83.57],
        'mpi3dhard': [71.09, 73.47, 70.88, 75.64],
        'celebahard': [68.24]
    },
    'CACTUS-DinoV2': {
        'shapes3d': [80.27, 80.40, 81.47, 80.35],
        'mpi3deasy': [94.76, 94.95, 92.87, 94.96],
        'mpi3dhard': [82.55, 80.63, 82.47, 82.04],
        'celebahard': [68.38]
    },
    'DRESS': {
        'shapes3d': [92.54, 93.15, 93.55, 92.97],
        'mpi3deasy': [99.89, 100.00, 99.89, 100.00],
        'mpi3dhard': [83.63, 84.38, 85.58, 86.22],
        'celebahard': [69.26]
    }
}


if __name__ == "__main__":
    print(f"Processing results...")

    latex_table = "\\toprule \n"
    latex_table += "Method & Shapes3D & MPI3D-Easy & MPI3D-Hard \\\\ \n\midrule \n"

    for method, res_dict in results.items():
        latex_table += f"{method} & "
        for ds in ['shapes3d', 'mpi3deasy', 'mpi3dhard', 'celebahard']:
            res_vals = res_dict[ds]
            if len(res_vals) == 0:
                latex_table += "TODO"
            else:
                res_avg = np.mean(res_vals)
                res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                latex_table += f"${res_avg:.2f}\% \pm {res_std:.2f}\%$ "
            if ds != "celebahard":
                latex_table += "&"
            else:
                latex_table += "\\\\ \n"
        if method in ["Supervised-Oracle", 
                      "Few-Shot Direct Adaptation", 
                      "Pre-Training and Fine-Tuning"]:
            latex_table += "\hline \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n" 
    
    # write the results to a file 
    latex_table_filename = "res_table.tex"
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")