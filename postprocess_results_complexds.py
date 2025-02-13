# Script to organize few-shot learning results of all the methods
# to a latex table file

import matplotlib.pyplot as plt
import os
from utils import *

results = {
    'Supervised-Original': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'Supervised-All': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'Supervised-Oracle': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'Few-Shot Direct Adaptation': {
        'celebahair': [62.93],
        'celebaprimary': [71.56],
        'celebarand': []
    },
    'Pre-Training and Fine-Tuning': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'Meta-GMVAE': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'PsCo':{
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'CACTUS-DeepCluster': {
        'celebahair': [],
        'celebaprimary': [],
        'celebarand': []
    },
    'CACTUS-DinoV2': {
        'celebahair': [69.85],
        'celebaprimary': [78.63],
        'celebarand': [79.90]
    },
    'DRESS': {
        'celebahair': [74.17],
        'celebaprimary': [75.22],
        'celebarand': [71.54]
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
                if method == "DRESS" and ds in ['celebahair']:
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
    latex_table_filename = "res_table_complexds.tex"
    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")