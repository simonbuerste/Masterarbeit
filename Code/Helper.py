"""
File for summarizing all helper Functions used for the
- Correct inputting of the Data
- evaluation of the Performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(label_list_test_merged, pred):
    """
    Plot of the Confusion Matrix for evaluation of Classification Performance per Class
    # Arguments
        :param label_list_test_merged:  Merged List of Labels for the Test Data
        :param pred:                    Predicted Labels from the Network --> Output of the L DNN Algorithm
    # Returns
        Nothing
    """

    # Create a Confusion Matrix for every Device (if more than one device is selected)
    for i in range(len(label_list_test_merged)):
        cm = confusion_matrix(y_true=np.asarray(label_list_test_merged[i]), y_pred=pred[i])
        cm = cm / cm.sum(axis=1)[:, None]

        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=np.unique(label_list_test_merged[i]),
               yticklabels=np.unique(label_list_test_merged[i]),
               title='Confusion Matrix of L DNN Algorithm for Index {}'.format(i))

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.mean()
        for k in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, k, format(cm[k, j], fmt),
                        ha="center", va="center",
                        color="white")
        fig.tight_layout()

    plt.show(block=False)


def data_separation(modul_a, no_classes, data_train, train_img_per_class, data_test, test_img_per_class):
    """
    Data is separated and the features are extracted for each class depending on the desired number of training/test
    images to incremental learning the different classes easily in the next pipeline steps
    # Arguments
        :param modul_a:                 Feature Extraction module
        :param no_classes:              The number of classes. For each class a list element is created
        :param data_train:              The Training dataset containing images and labels
        :param train_img_per_class:     Desired number of training images per class which will be separated (Randomly shuffling)
        :param data_test:               The Test dataset containing images and labels
        :param test_img_per_class:      Desired number of training images per class which will be separated (Randomly shuffling)
    # Returns
        feature_list_train:             The extracted Features from training data separated by each class
        label_list_train:               The corresponding Labels to the training data
        feature_list_test:              The extracted Features from test data separated by each class
        label_list_test:                The corresponding Labels to the test data
    """

    # To seperate the data correct, create a nested list with seperat lists for every class
    feature_list_train = [[] for j in range(no_classes)]
    feature_list_test = [[] for j in range(no_classes)]
    label_list_train = [[] for j in range(no_classes)]
    label_list_test = [[] for j in range(no_classes)]
    i = np.zeros((no_classes,), dtype=int)

    # Loop over all training data (or until break condition is fullfilled)
    for img, label in data_train:
        # Break if for every class enough training samples are stored already
        all_data = i >= train_img_per_class
        if np.all(all_data == True):
            break

        # Just add the data to the list if not enough data for this class is stored yet
        label_scalar = (np.array(label)).item()
        if i[label_scalar] < train_img_per_class:
            features = modul_a.predict(img, steps=1)
            features_np = np.array(features)
            features_np_normed = features_np/np.sum(features_np)
            feature_list_train[label_scalar].append(features_np_normed.flatten())
            label_list_train[label_scalar].append(label)
            i[label_scalar] += 1

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

    return feature_list_train, label_list_train, feature_list_test, label_list_test


def accuracy_plot(accuracy, no_classes, no_groups, no_edge_devices):
    """
    The Accuracy over the number of classes the system has learned is plotted to show the behaviour of the algorithm
    with respect to catastrophic forgetting
    # Arguments
        :param accuracy:            Accuracy of every edge device (seperated row-wise)
        :param no_classes:          Number of classes in the dataset
        :param no_groups:           Number of groups the dataset is clustered into
        :param no_edge_devices:     The number of edge devices in this scenario
    # Returns
        Nothing
    """
    fig, ax = plt.subplots()
    x_axis = np.arange(no_classes/no_groups, (no_classes/no_edge_devices)+1, no_classes/no_groups)
    ax.set(title='Accuracy over Number of Classes',
           xticks=x_axis,
           yticks=np.arange(0, 1, 0.1),
           xlim=[np.min(x_axis), np.max(x_axis)],
           ylim=[0, 1],
           xlabel='Number of Classes',
           ylabel='Classification Accuracy')
    ax.plot(x_axis, accuracy, '-o')
    ax.grid()
    fig.tight_layout()
    plt.show()
