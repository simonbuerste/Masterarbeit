import os
import json

import numpy as np
import matplotlib.pyplot as plt

directory = ["C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/2019-09-23_09-27-47_No_Groups_10",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/iCarl",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/lwf",
             "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012/2019-09-26_08-36-11_No_Groups_20"]

accuracy = np.zeros((len(directory), 10))

for i, dir in enumerate(directory):
    filename_metrics = os.path.join(dir, 'metrics.json')

    with open(filename_metrics, "r") as eval_metrics_file:
        eval_metrics = json.load(eval_metrics_file)

    if len(eval_metrics["Accuracy"]) > 10:
        accuracy_different_scale = np.asarray(eval_metrics["Accuracy"]).flatten()
    else:
        accuracy[i, :] = np.asarray(eval_metrics["Accuracy"]).flatten()

save_dir = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet2012"
title = "Klassifikations Genauigkeit bei ImageNet"
y_label = "Klassifikations Genauigkeit in %"
x_label = "Anzahl an trainierten Klassen"
save_name = "Accuracy_imagenet"
param_values = np.arange(100, 1100, 100)
param_values_different_scale = np.arange(50, 1050, 50)
fig, ax = plt.subplots()
for i in range(len(directory)-1):
    ax.plot(param_values, accuracy[i, :], linestyle='-', marker='o')
ax.set(xlabel=x_label,
       ylabel=y_label,
       xticks=param_values,
       yticks=np.arange(0, 1.1, 0.1),
       ylim=[0, 1],
       title=title)
ax.plot(param_values_different_scale, accuracy_different_scale, linestyle='-', marker='o')
ax.legend(["L DNN 10 Schritte", "iCaRL", "LwF", "L DNN 20 Schritte"])
ax.grid(True)
plt.tight_layout()
filename = os.path.join(save_dir, save_name+"_Line_Plot")
plt.savefig(filename + ".png")
plt.savefig(filename + ".svg")
plt.show()
