"""
File for summarizing all helper Functions used for the
- Separation of the Data depending on their classes
- evaluation and visualization of the Performance
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def data_separation_train(modul_a, data_train, params):
    """
    Data is separated and the features are extracted for each class depending on the desired number of training/test
    images to incremental learning the different classes easily in the next pipeline steps
    # Arguments
        :param modul_a:         Feature Extraction module
        :param data_train:      The Training dataset containing images and labels
        :param params           Dictionary containing all parameters
    # Returns
        feature_list_train:     The extracted Features from training data separated by each class
        label_list_train:       The corresponding Labels to the training data
    """
    print(tf.executing_eagerly())
    
    train_img_per_class = params["train_img_per_class"]
    no_classes = params["no_classes"]

    # To seperate the data correct, create a nested list with seperat lists for every class
    feature_list_train = [[] for _ in range(no_classes)]
    label_list_train = [[] for _ in range(no_classes)]
    i = np.zeros((no_classes,), dtype=int)

    # Loop over all training data (or until break condition is fullfilled)
    for img, label in data_train:
        # Break if for every class enough training samples are stored already
        all_data = i >= train_img_per_class
        if np.all(all_data == True):
            break

        # Just add the data to the list if not enough data for this class is stored yet
        # print(tf.executing_eagerly())
        label_scalar = (np.array(label)).item()
        if i[label_scalar] < train_img_per_class:
            features = modul_a.predict(img, steps=1)
            features_np = np.array(features)
            features_np_normed = features_np/np.sum(features_np)
            feature_list_train[label_scalar].append(features_np_normed.flatten())
            label_list_train[label_scalar].append(label)
            i[label_scalar] += 1

    return feature_list_train, label_list_train


def data_separation_test(modul_a, data_test, params):
    """
    Data is separated and the features are extracted for each class depending on the desired number of training/test
    images to incremental learning the different classes easily in the next pipeline steps
    # Arguments
        :param modul_a:         Feature Extraction module
        :param data_test:       The Test dataset containing images and labels
        :param params           Dictionary containing all parameters
    # Returns
        feature_list_test:      The extracted Features from test data separated by each class
        label_list_test:        The corresponding Labels to the test data
    """

    test_img_per_class = params["test_img_per_class"]
    no_classes = params["no_classes"]

    # To seperate the data correct, create a nested list with seperat lists for every class
    feature_list_test = [[] for _ in range(no_classes)]
    label_list_test = [[] for _ in range(no_classes)]
    i = np.zeros((no_classes,), dtype=int)

    # Loop over all test data (or until break condition is fullfilled)
    for img, label in data_test:
        # Break if for every class enough training samples are stored already
        all_data = i >= test_img_per_class
        if np.all(all_data == True):
            break

        # Just add the data to the list if not enough data for this class is stored yet
        label_scalar = (np.array(label)).item()
        if i[label_scalar] < test_img_per_class:
            features = modul_a.predict(img, steps=1)
            features_np = np.array(features)
            features_np_normed = features_np/np.sum(features_np)
            feature_list_test[label_scalar].append(features_np_normed.flatten())
            label_list_test[label_scalar].append(label)
            i[label_scalar] += 1

    return feature_list_test, label_list_test


def confusion_matrix_plot(label_list_test_merged, pred, params):
    """
    Plot of the Confusion Matrix for evaluation of Classification Performance per Class
    # Arguments
        :param label_list_test_merged:  Merged List of Labels for the Test Data
        :param pred:                    Predicted Labels from the Network --> Output of the L DNN Algorithm
        :param params:                  Dictionary containing all parameters
    # Returns
        Nothing
    """

    # Create a Confusion Matrix for every Device (if more than one device is selected)
    for i in range(len(label_list_test_merged)):
        cm = np.array(tf.math.confusion_matrix(labels=np.asarray(label_list_test_merged[i]), predictions=pred[i]))
        cm = cm / cm.sum(axis=1)[:, None]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=np.unique(label_list_test_merged[i]),
               yticklabels=np.unique(label_list_test_merged[i]),
               title='Confusion Matrix of L DNN Algorithm for Device {}'.format(i+1))
        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2
        for k in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, k, format(cm[k, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[k, j] > thresh else "black")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        fig.tight_layout()

        # Save Figures to given directory
        if params["save"] is True:
            filename = os.path.join(params["dir2save"], "Confusion_Matrix_Device_{}".format(i+1))
            plt.savefig(filename+".png")
            plt.savefig(filename+".svg")

    # plt.show(block=False)


def accuracy_plot(accuracy, params):
    """
    The Accuracy over the number of classes the system has learned is plotted to show the behaviour of the algorithm
    with respect to catastrophic forgetting
    # Arguments
        :param accuracy:    Accuracy of every edge device (seperated row-wise)
        :param params:      Dictionary containing all parameters
    # Returns
        Nothing
    """
    no_classes = params["no_classes"]
    no_groups = params["no_groups"]
    no_edge_devices = params["no_edge_devices"]

    fig, ax = plt.subplots()
    x_axis = np.arange(no_classes/no_groups, (no_classes/no_edge_devices)+1, no_classes/no_groups)
    ax.set(title='Accuracy over Number of Classes',
           xticks=x_axis,
           yticks=np.arange(0, 1, 0.1),
           xlim=[np.min(x_axis), np.max(x_axis)],
           ylim=[0, 1])
    ax.set_xlabel("Number of Classes", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    ax.plot(x_axis, accuracy, '-o')
    ax.grid()
    fig.tight_layout()

    # Save Figures to given directory
    if params["save"] is True:
        filename = os.path.join(params["dir2save"], "Accuracy_Plot")
        plt.savefig(filename+".png")
        plt.savefig(filename+".svg")

    # plt.show()
    plt.close('all')
