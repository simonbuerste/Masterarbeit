import os
import json

import numpy as np
import matplotlib.pyplot as plt

directory = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/rho_alpha/mnist"
list_dir = next(os.walk(directory))[1]
summary = []

for i, result_folder in enumerate(list_dir):
    filename_metrics = os.path.join(directory, result_folder, 'metrics.json')
    filename_params = os.path.join(directory, result_folder, 'params.json')

    with open(filename_metrics, "r") as eval_metrics_file:
        eval_metrics = json.load(eval_metrics_file)

    with open(filename_params, "r") as eval_params_file:
        eval_params = json.load(eval_params_file)

    tmp = {
        "Accuracy_Final":   np.asarray(eval_metrics["Accuracy"])[-1],
        "Memory Modul A":   eval_metrics["Memory Module A"],
        "Memory Modul B":   eval_metrics["Memory Module B"],
        "Alpha":            eval_params["modul_b_alpha"],
        "Rho":              eval_params["modul_b_rho"]
    }

    if i == 0:
        repetitions = eval_params["no_repetitions"]

    summary.append(tmp)

alpha_values = np.unique([d["Alpha"] for d in summary])
rho_values = np.unique([d["Rho"] for d in summary])

confusion_matrix = np.zeros((len(alpha_values), len(rho_values)))

for d in summary:
    idx0 = np.where(alpha_values == d["Alpha"])
    idx1 = np.where(rho_values == d["Rho"])
    confusion_matrix[idx0, idx1] += d["Accuracy_Final"]

confusion_matrix = confusion_matrix / repetitions

# ----- Confusion Matrix Plot -----
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(confusion_matrix.shape[1]),
       yticks=np.arange(confusion_matrix.shape[0]),
       xticklabels=rho_values, yticklabels=alpha_values,
       title='Confusion Matrix for different Hyperparameter Values',
       xlabel='Rho', ylabel='Alpha')
ax.set_xticks(np.arange(confusion_matrix.shape[1] + 1) - .5, minor=True)
ax.set_yticks(np.arange(confusion_matrix.shape[0] + 1) - .5, minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = confusion_matrix.mean()
for k in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(j, k, format(confusion_matrix[k, j], fmt),
                ha="center", va="center",
                color="white" if confusion_matrix[k, j] > thresh else "black")
fig.tight_layout()
filename = os.path.join(directory, "Confusion_Matrix_Grid_Search")
plt.savefig(filename + ".png")
plt.savefig(filename + ".svg")

plt.show()
