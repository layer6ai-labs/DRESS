# Script to organize few-shot learning results of all the methods
# to a latex table file

import os
import sys
sys.path.append("../")
from utils import *

results = {
    'Supervised-Original': {
        'celebahair': [67.95],
        'celebaprimary': [73.26],
        'celebarand': [81.28]
    },
    'Supervised-All': {
        'celebahair': [79.74],
        'celebaprimary': [87.62],
        'celebarand': [85.33]
    },
    'Supervised-Oracle': {
        'celebahair': [87.07],
        'celebaprimary': [91.08],
        'celebarand': [90.57]
    },
    'Few-Shot Direct Adaptation': {
        'celebahair': [64.00],
        'celebaprimary': [69.12],
        'celebarand': [57.82]
    },
    'Pre-Training and Fine-Tuning': {
        'celebahair': [60.01],
        'celebaprimary': [66.49],
        'celebarand': [64.38]
    },
    'Meta-GMVAE': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'PsCo':{
        'celebahair': [61.90],
        'celebaprimary': [66.25],
        'celebarand': [59.34]
    },
    'CACTUS-DeepCluster': {
        'celebahair': [70.57],
        'celebaprimary': [71.59],
        'celebarand': [65.87]
    },
    'CACTUS-DinoV2': {
        'celebahair': [69.85, 69.11, 69.63, 68.88],
        'celebaprimary': [77.04, 77.48, 77.50, 76.00],
        'celebarand': [74.59, 74.42, 75.00, 73.54]
    },
    'DRESS': {
        'celebahair': [73.86, 74.18, 73.87],
        'celebaprimary': [77.12, 77.40, 77.77],
        'celebarand': [70.01, 67.49, 67.97]
    }
}


if __name__ == "__main__":
    print(f"Processing results...")

    latex_table = "\\toprule \n"
    latex_table += "Method & CelebA-Hair & CelebA-Primary & CelebA-Random \\\\ \n\midrule \n"

    for method, res_dict in results.items():
        if method == 'Few-Shot Direct Adaptation':
            latex_table += "\\makecell[l]{Few-Shot Direct \\\\ Adaptation} & "
        elif method == 'Pre-Training and Fine-Tuning':
            latex_table += "\\makecell[l]{Pre-Training \\\\ and Fine-Tuning} & "
        elif method == 'DRESS':
            latex_table += "\\textbf{DRESS} & "
        else:
            latex_table += f"{method} & "
        for ds in ['celebahair', 'celebaprimary', 'celebarand']:
            res_vals = res_dict[ds]
            if len(res_vals) == 0:
                latex_table += "TODO"
            else:
                res_avg = np.mean(res_vals)
                res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                if (method == "DRESS" and ds in ['celebahair', 'celebaprimary']) or \
                    (method == "CACTUS-DinoV2" and ds in ['celebarand']):
                    # bold font
                    latex_table += f"\\textbf{{{res_avg:.2f}}}\% $\pm$ \\textbf{{{res_std:.2f}}}\%"
                else:
                    latex_table += f"${res_avg:.2f}\% \pm {res_std:.2f}\%$"
            if ds != "celebarand":
                latex_table += " & "
            else:
                latex_table += "\\\\ \n"
        if method in ["Supervised-Oracle", 
                      "Few-Shot Direct Adaptation", 
                      "PsCo"]:
            latex_table += "\hline \n"
    # finishing the table by an underline
    latex_table += "\\bottomrule \n" 
    
    # write the results to a file 
    latex_table_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "res_table_complexds.tex")
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")