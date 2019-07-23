"""
This file is the Main File for executing the Lifelong DNN Algorithm
Pipeline is set up here and all relevant Parameters can be set here
"""

from Data_Input import input_fn
from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP
from Continual_Learning import continual_learning_fun
from Helper import confusion_matrix_plot
from Helper import data_separation

# Parameters for the Pipeline are set here
no_classes = 10
no_groups = 10
train_img_per_class = 5
test_img_per_class = 10

# Input Function is called which provides the training and test data for the desired Dataset
data_train, data_test = input_fn(dataset="mnist", visu=True)

# The Feature Extraction Module A is called and created/downloaded with the corresponding image size
modulA = modul_a(image_size=96)

# Data is separated per Class and the features are extracted with Module A
feature_list_train, label_list_train, feature_list_test, label_list_test = data_separation(modulA, no_classes,
                                                                                           data_train,
                                                                                           train_img_per_class,
                                                                                           data_test,
                                                                                           test_img_per_class)

# Module B is initialized with the corresponding parameters
modulB = FuzzyARTMAP(alpha=0.25, rho=0.65, n_classes=no_classes, s=1.05)

# The Training and Testing of the L DNN Algorithm takes place according to the previously defined and created
# Modules and Data.
label_list_test_merged, pred = continual_learning_fun(modulB, no_classes, no_groups, feature_list_train, label_list_train,
                                                      feature_list_test, label_list_test)

# For a final Evaluation a Confusion Matrix is plotted
confusion_matrix_plot(label_list_test_merged, pred)
