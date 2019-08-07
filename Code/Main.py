"""
This file is the Main File for executing the Lifelong DNN Algorithm.
The Pipeline is set up here and all relevant Parameters can be set here
"""

import os
import json

from time import gmtime, strftime
from pympler import asizeof

from Data_Input import input_fn
from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP
from Continual_Learning import continual_learning_fun
from Distributed_Learning import distributed_learning_fun
from Helper import confusion_matrix_plot
from Helper import data_separation
from Helper import accuracy_plot

# Parameters for the Pipeline are set here

params = {
    'save':                 True,
    'dir2save':             'C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/',
    'no_classes':           10,
    'no_groups':            8,
    'train_img_per_class':  5,
    'test_img_per_class':   10,
    'no_edge_devices':      1,
    'test_case':            "continual"  # "distributed"#
}


if params["save"] is True:
    # get current time as string for saving of model
    timestring = strftime("%Y-%m-%d_%H-%M", gmtime())
    params["dir2save"] = os.path.join(params["dir2save"], timestring)
    # Create directory for model combination if not existens
    if not os.path.exists(params["dir2save"]):
        os.makedirs(params["dir2save"])
    # Write Parameters to Json to check impact of parameter values on results
    filename_params = os.path.join(params["dir2save"], "params.json")
    with open(filename_params, "w") as params_file:
        json.dump(params, params_file, sort_keys=True)

# Input Function is called which provides the training and test data for the desired Dataset
data_train, data_test = input_fn(dataset="imagenet10", visu=False)

# The Feature Extraction Module A is called and created/downloaded with the corresponding image size
modulA = modul_a(image_size=96)

# Data is separated per Class and the features are extracted with Module A
feature_list_train, label_list_train, feature_list_test, label_list_test = data_separation(modulA, data_train,
                                                                                           data_test, params)

# Module B is initialized with the corresponding parameters
modulB = [[] for i in range(params["no_edge_devices"])]
for i in range(params["no_edge_devices"]):
    modulB[i] = FuzzyARTMAP(alpha=0.25, rho=0.5, n_classes=params["no_classes"], s=1.05, epsilon=0.001)

# Training and Testing of the L DNN Algorithm according to the previously defined and created Modules and Data.
if params["test_case"] == "continual":
    label_list_test_merged, pred, accuracy = continual_learning_fun(modulB[0], params["no_classes"],
                                                                    params["no_groups"], feature_list_train,
                                                                    label_list_train, feature_list_test,
                                                                    label_list_test)
elif params["test_case"] == "distributed":
    label_list_test_merged, pred, accuracy, accuracy_melded = distributed_learning_fun(modulB, params["no_classes"],
                                                                                       params["no_groups"],
                                                                                       feature_list_train,
                                                                                       label_list_train,
                                                                                       feature_list_test,
                                                                                       label_list_test)
else:
    print("Test Case not available or wrongly written. Please choose between 'continual' or 'distributed'.")
    exit()

# For a Evaluation Plots are generated
confusion_matrix_plot(label_list_test_merged, pred, params)
accuracy_plot(accuracy, params)

# Get Memory Consumption of the modules
mem_modulA = asizeof.asizeof(modulA)
mem_modulB = asizeof.asizeof(modulB[0])
print("Memory usage of Module A: {} KB".format(mem_modulA/1000))
print("Memory usage of Module B: {} KB".format(mem_modulB/1000))

if params["save"] is True:
    metrics = {
        'Memory Module A':  mem_modulA,
        'Memory Module B':  mem_modulB,
        'Accuracy':         accuracy.tolist()
    }
    if 'accuracy_melded' in locals():
        metrics["Accuracy melded"] = accuracy_melded

    # Write Parameters to Json to check impact of parameter values on results
    filename_metrics = os.path.join(params["dir2save"], "metrics.json")
    with open(filename_metrics, "w") as metrics_file:
        json.dump(metrics, metrics_file, sort_keys=True)
