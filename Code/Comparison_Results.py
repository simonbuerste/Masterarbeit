import os
import json

import numpy as np
import matplotlib.pyplot as plt

directory = ["C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/2019-09-23_09-27-47_No_Groups_10",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/iCarl",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/lwf"]

accuracy = np.zeros((len(directory), 10))

for i, dir in enumerate(directory):
    filename_metrics = os.path.join(dir, 'metrics.json')

    with open(filename_metrics, "r") as eval_metrics_file:
        eval_metrics = json.load(eval_metrics_file)

    accuracy[i, :] = np.asarray(eval_metrics["Accuracy"]).flatten()

save_dir = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012"
title = "Klassifikations Genauigkeit bei ImageNet"
y_label = "Klassifikations Genauigkeit in %"
x_label = "Anzahl an trainierten Klassen"
save_name = "Accuracy_imagenet"
param_values = np.arange(100, 1100, 100)

fig, ax = plt.subplots()
for i in range(len(directory)):
    ax.plot(param_values, accuracy[i, :], linestyle='-', marker='o')
ax.set(xlabel=x_label,
       ylabel=y_label,
       xticks=param_values,
       yticks=np.arange(0, 1.1, 0.1),
       ylim=[0, 1],
       title=title)
ax.legend(["L DNN", "iCaRL", "LwF"])
ax.grid(True)
plt.tight_layout()
filename = os.path.join(save_dir, save_name+"_Line_Plot")
plt.savefig(filename + ".png")
plt.savefig(filename + ".svg")
plt.show()
