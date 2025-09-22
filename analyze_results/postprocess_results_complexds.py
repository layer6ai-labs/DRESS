# Script to organize few-shot learning results of all the methods
# to a latex table file

import os
import sys
sys.path.append("../")
from utils import *
from res_all import ACCURACIES_ALL

if __name__ == "__main__":
    print(f"Processing results...")

    latex_table = "\\toprule \n"
    latex_table += "\\multirow{2}{*}{Method} & "
    latex_table += "\\multicolumn{2}{l}{CelebA-Hair} & "
    latex_table += "\\multicolumn{2}{l}{CelebA-Primary} & "
    latex_table += "\\multicolumn{2}{l}{CelebA-Random} \\\\ \n"
    latex_table += " & 5-Shot & 10-Shot & 5-Shot & 10-Shot & 5-Shot & 10-Shot \\\\ \n"
    latex_table += "\\midrule \n"

    for method, res_dict in ACCURACIES_ALL.items():
        if method.startswith("Ablate"):
            continue
        if method == 'Few-Shot Direct Adaptation':
            latex_table += "FSDA & "
        elif method == 'Pre-Training and Fine-Tuning':
            latex_table += "PTFT & "
        elif method == 'DRESS':
            latex_table += "\\textbf{DRESS} & "
        else:
            latex_table += f"{method} & "
        for ds in ['celebahair', 'celebaprimary', 'celebarand']:
            for shot in ['five-shot', "ten-shot"]:
                res_vals = res_dict[shot][ds]
                if len(res_vals) == 0:
                    latex_table += "TODO"
                else:
                    res_avg = np.mean(res_vals)
                    res_std = np.std(res_vals) / np.sqrt(len(res_vals))
                    if (method == "DRESS" and ds in ['celebahair', 'celebaprimary']) or \
                        (method == "CACTUS-DINO" and ds in ['celebarand']):
                        # bold font
                        latex_table += f"\\makecell[l]{{\\textbf{{{res_avg:.1f}}}\% \\\\ {{\\tiny $\pm$ \\textbf{{{res_std:.1f}}}}}\%}}"
                    else:
                        latex_table += f"\\makecell[l]{{${res_avg:.1f}\%$ \\\\ {{\\tiny $\pm {res_std:.1f}\%$}}}}"
                if ds != "celebarand":
                    latex_table += " & "
                else:
                    if shot == "five-shot":
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