import os
import json

import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
directory = ["C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/Consolidation/Every/split-mnist",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/Consolidation/Final/split-mnist",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/split-mnist"]

accuracy = np.zeros((len(directory), 5))

for i, dir in enumerate(directory):
    filename_metrics = os.path.join(dir, 'metrics_merged.json')
=======
directory = ["C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/2019-09-23_09-27-47_No_Groups_10",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/iCarl",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/lwf"]

accuracy = np.zeros((len(directory), 10))

for i, dir in enumerate(directory):
    filename_metrics = os.path.join(dir, 'metrics.json')
>>>>>>> 4b748df04afb6158d4f1b10b3a28d3791c0ef2a9

    with open(filename_metrics, "r") as eval_metrics_file:
        eval_metrics = json.load(eval_metrics_file)

<<<<<<< HEAD
    accuracy[i, :] = np.asarray(eval_metrics["Accuracy Training Steps"])

save_dir = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/"
title = "Klassifikations Genauigkeit bei Split-MNIST"
y_label = "Klassifikations Genauigkeit in %"
x_label = "Anzahl an trainierten Klassen"
save_name = "Accuracy_mnist"
param_values = np.arange(2, 12, 2)
=======
    accuracy[i, :] = np.asarray(eval_metrics["Accuracy"]).flatten()

save_dir = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012"
title = "Klassifikations Genauigkeit bei ImageNet"
y_label = "Klassifikations Genauigkeit in %"
x_label = "Anzahl an trainierten Klassen"
save_name = "Accuracy_imagenet"
param_values = np.arange(100, 1100, 100)
>>>>>>> 4b748df04afb6158d4f1b10b3a28d3791c0ef2a9

fig, ax = plt.subplots()
for i in range(len(directory)):
    ax.plot(param_values, accuracy[i, :], linestyle='-', marker='o')
ax.set(xlabel=x_label,
       ylabel=y_label,
       xticks=param_values,
       yticks=np.arange(0, 1.1, 0.1),
       ylim=[0, 1],
       title=title)
<<<<<<< HEAD
ax.legend(["Konsolidierung jeder Schritt", "Konsolidierung finaler Schritt", "Keine Konsolidierung"])
=======
ax.legend(["L DNN", "iCaRL", "LwF"])
>>>>>>> 4b748df04afb6158d4f1b10b3a28d3791c0ef2a9
ax.grid(True)
plt.tight_layout()
filename = os.path.join(save_dir, save_name+"_Line_Plot")
plt.savefig(filename + ".png")
plt.savefig(filename + ".svg")
plt.show()
