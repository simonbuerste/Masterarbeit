import os
import json

import numpy as np
import matplotlib.pyplot as plt

directory = "C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet10"
list_dir = next(os.walk(directory))[1]
summary = []

hyperparameters = ["No_Train_Image_per_Class"]  # ["Alpha", "Rho"]

for i, result_folder in enumerate(list_dir):
    filename_metrics = os.path.join(directory, result_folder, 'metrics.json')
    filename_params = os.path.join(directory, result_folder, 'params.json')

    with open(filename_metrics, "r") as eval_metrics_file:
        eval_metrics = json.load(eval_metrics_file)

    with open(filename_params, "r") as eval_params_file:
        eval_params = json.load(eval_params_file)

    tmp = {
        "Accuracy_Final":           np.asarray(eval_metrics["Accuracy"])[-1],
        "Memory Modul A":           eval_metrics["Memory Module A"],
        "Memory Modul B":           eval_metrics["Memory Module B"],
        "Alpha":                    eval_params["modul_b_alpha"],
        "Rho":                      eval_params["modul_b_rho"],
        "No_Train_Image_per_Class": eval_params["train_img_per_class"]
    }

    if i == 0:
        repetitions = eval_params["no_repetitions"]

    summary.append(tmp)

if len(hyperparameters) == 1:
    param_values = np.unique([d[hyperparameters[0]] for d in summary])

    idx_counter = np.zeros((len(param_values),), dtype=int)
    plot_values = np.zeros((len(param_values), repetitions))
    for d in summary:
        idx = np.where(param_values == d[hyperparameters[0]])
        idx_counter[idx] += 1
        # plot_values[idx] += d["Memory Modul B"]/1000000
        plot_values[idx, idx_counter[idx]-1] += d["Accuracy_Final"]

    mu = np.mean(plot_values, axis=1)
    std = np.std(plot_values, axis=1)
    title = "Classification Accuracy on ImageNet-10"
    # title = "Classification Accuracy on ImageNet-10"
    y_label = "Classification Accuracy in %"
    # y_label = "Memory Consumption Module B in MB"
    save_name = "Accuracy"
    # save_name = "Memory_Module_B"

    fig, ax = plt.subplots()
    ax.plot(param_values, mu, linestyle='-', marker='o')
    ax.set(#xscale='log',
           xlabel='%s' % hyperparameters[0],
           ylabel=y_label,
           yticks=np.arange(0, 1.1, 0.1),
           ylim=[0, 1],
           title=title)
    ax.grid(True)
    plt.tight_layout()
    filename = os.path.join(directory, save_name+"_Line_Plot")
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".svg")
    plt.show()

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(param_values, mu, yerr=std, align='center', ec="k", ecolor='black', capsize=5, width=0.25*np.array(param_values))
    ax.set(#xscale='log',
           xlabel='%s' % hyperparameters[0],
           ylabel=y_label,
           yticks=np.arange(0, 1.1, 0.1),
           ylim=[0, 1],
           title=title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    filename = os.path.join(directory, save_name+"_bar_plot_with_error_bars")
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".svg")
    plt.show()

    # Summarize Metrics over multiple runs
    metrics_merged = {
        "parameter_values": param_values.tolist(),
        "Accuracy Mean":    mu.tolist(),
        "Accuracy Std":     std.tolist()
    }

    # Write Parameters to Json to check impact of parameter values on results
    filename_metrics_merged = os.path.join(directory, "metrics_merged.json")
    with open(filename_metrics_merged, "w") as metrics_file:
        json.dump(metrics_merged, metrics_file, sort_keys=True)

elif len(hyperparameters) == 2:
    alpha_values = np.unique([d["Alpha"] for d in summary])
    rho_values = np.unique([d["Rho"] for d in summary])

    confusion_matrix = np.zeros((len(alpha_values), len(rho_values)))

    for d in summary:
        idx0 = np.where(alpha_values == d["Alpha"])
        idx1 = np.where(rho_values == d["Rho"])
        confusion_matrix[idx0, idx1] += d["Accuracy_Final"]# d["Memory Modul B"]/1000000

    confusion_matrix = confusion_matrix / repetitions

    # ----- Confusion Matrix Plot -----
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=np.around(rho_values, decimals=1), yticklabels=np.around(alpha_values, decimals=1),
           # title='Memory Module B in MB',
           title='Classification Accuracy in %',
           xlabel='Rho', ylabel='Alpha')
    ax.set_xticks(np.arange(confusion_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(confusion_matrix.shape[0] + 1) - .5, minor=True)
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = confusion_matrix.mean()
    for k in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, k, format(confusion_matrix[k, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[k, j] > thresh else "black")
    fig.tight_layout()
    # filename = os.path.join(directory, "Memory_Module_B_Grid_Search")
    filename = os.path.join(directory, "Confusion_Matrix_Grid_Search")
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".svg")

    plt.show()
