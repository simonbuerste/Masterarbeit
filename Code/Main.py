"""
This file is the Main File for executing the Lifelong DNN Algorithm.
The Pipeline is set up here and all relevant Parameters can be set here
"""

import os
import json
import numpy as np

from time import gmtime, strftime
from pympler import asizeof

from Data_Input import input_fn
from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP
from Continual_Learning import continual_learning_fun
from Distributed_Learning import distributed_learning_fun
from Helper import confusion_matrix_plot
from Helper import data_separation_train
from Helper import data_separation_test
from Helper import accuracy_plot

# Parameters for the Pipeline are set here
params = {
    'save':                 True,
    'dir2save':             'C:/Users/simon/Documents/Uni_Stuttgart/Masterarbeit/Results/CL_Final/imagenet10_full',
    'dataset':              "imagenet10",
    'no_classes':           10,
    'no_groups':            1,
    'train_img_per_class':  1,
    'test_img_per_class':   50,
    'no_edge_devices':      1,
    'test_case':            "continual",  # "distributed"#
    'no_repetitions':       5,
    'modul_b_alpha':        0.2,
    'modul_b_rho':          0.5,
    'modul_b_s':            1.05,
    'modul_b_epsilon':      0.001
}

# The Feature Extraction Module A is called and created/downloaded with the corresponding image size
modulA = modul_a(image_size=96)

# Data is separated per Class and the features are extracted with Module A
# Done one time for the test data to have same test data for evaluation
# feature_list_test, label_list_test = data_separation_test(modulA, data_test, params)

dir_orig = params["dir2save"]

no_train_img = [100]
# Nested for-loops for Hyperparameter Optimization
for a in no_train_img:
    params["train_img_per_class"] = a
    # For-loop for number of repetitions to get reliable results
    for _ in range(params["no_repetitions"]):
        if params["save"] is True:
            # get current time as string for saving of model
            timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) # + "_No_Train_Img_{}".format(a)
            params["dir2save"] = os.path.join(dir_orig, timestring)
            # Create directory for model combination if not existens
            if not os.path.exists(params["dir2save"]):
                os.makedirs(params["dir2save"])
            # Write Parameters to Json to check impact of parameter values on results
            filename_params = os.path.join(params["dir2save"], "params.json")
            with open(filename_params, "w") as params_file:
                json.dump(params, params_file, sort_keys=True)

        # Input Function is called which provides the training and test data for the desired Dataset
        data_train, data_test = input_fn(dataset=params["dataset"], visu=False)

        # Data is separated per Class and the features are extracted with Module A
        # Repeat for every repetition for the training data to see impact of choice of training data
        feature_list_train, label_list_train = data_separation_train(modulA, data_train, params)

        # Test List is filled with Validation Data. Cross-Validation is used with randomly splitting between
        # Training and Validation Data. Training Dataset is used for this.
        feature_list_test, label_list_test = data_separation_test(modulA, data_test, params)

        # Module B is initialized with the corresponding parameters
        modulB = [[] for i in range(params["no_edge_devices"])]
        for i in range(params["no_edge_devices"]):
            modulB[i] = FuzzyARTMAP(alpha=params["modul_b_alpha"], rho=params["modul_b_rho"],
                                    n_classes=params["no_classes"], s=params["modul_b_s"],
                                    epsilon=params["modul_b_epsilon"])

        # Training and Testing of the L DNN Algorithm according to previously defined and created Modules and Data
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

        # Evaluation Plots are generated
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
