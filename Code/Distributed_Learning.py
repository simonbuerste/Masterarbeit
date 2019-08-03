import numpy as np
import random
import copy

"""
Function for the distributed incremental class learning is implemented here
"""


def distributed_learning_fun(modul_b, no_classes, no_groups, feature_list_train, label_list_train, feature_list_test,
                             label_list_test):
    """
    Distributed (incremental) Class Learning is executed here. The classes are learned one after each other at the
    different "devices" (here: Different instances of the FuzzyARTMAP class) and testing on all previously learned
    classes is done.
    # Arguments
        :param modul_b:             The created Module B which is in the L DNN Algorithm the incremental Classifier.
                                    Class Object with defined Functions and Parameters
        :param no_classes:          Number of classes in the dataset
        :param no_groups:           Number of Training Groups. Can be used if more classes should be trained together
        :param feature_list_train:  The extracted Features from training data separated by each class
        :param label_list_train:    The corresponding Labels to the training data
        :param feature_list_test:   The extracted Features from test data separated by each class
        :param label_list_test:     The corresponding Labels to the training data
    # Returns
        label_list_test_merged:     The merged List of Labels for the test data
        pred:                       Predicted Labels from the Network --> Output of the L DNN Algorithm
        accuracy:                   Classification Accuracy for the different incremental Learning Steps
    """

    # Create Lists for Test Data and Prediction for defined number of Devices (Modules) plus one melded Module
    feature_list_test_merged = [[] for _ in range(len(modul_b)+1)]
    label_list_test_merged = [[] for _ in range(len(modul_b)+1)]
    pred = [[] for _ in range(len(modul_b)+1)]

    # Random shuffle the classes for reliable Results with more repetitions
    classes = np.arange(0, no_classes)
    random.shuffle(classes)
    print(classes)

    group_splitting = int(np.ceil(no_groups/len(modul_b)))
    accuracy = np.zeros((group_splitting, len(modul_b)))

    edge_device_idx = 0

    for i in range(no_groups):  # - 1, -1, -1):
        # When Groups are splitted between multiple devices and not the first device is selected,
        # edge device index is incremented. Additionally save old trained Module and take the next module for training
        if i % group_splitting == 0:
            if i > 0:
                modul_b[edge_device_idx] = modul_b_tmp
                edge_device_idx += 1
            modul_b_tmp = modul_b[edge_device_idx]

        print("------ Training Group {} of {} at Edge Device {} ------".format(int(i+1), no_groups, edge_device_idx))

        group_size = int(no_classes/no_groups)
        idx = i*group_size

        tmp_features = []
        tmp_labels = []
        # Get all training and test data from the Group. A group may contain more than one class (defined by group_size)
        for j in range(idx, (idx+group_size)):
            tmp_features = tmp_features + feature_list_train[classes[j]]
            tmp_labels = tmp_labels + label_list_train[classes[j]]

            feature_list_test_merged[edge_device_idx] = feature_list_test_merged[edge_device_idx] + feature_list_test[classes[j]]
            label_list_test_merged[edge_device_idx] = label_list_test_merged[edge_device_idx] + label_list_test[classes[j]]

        # A Group can be selected which should not be trained
        if i != 10:
            modul_b_tmp.train(np.array(tmp_features), np.asarray(tmp_labels)[:, 0], epochs=1)

        # If desired, Consolidation of Weights for all classes can be done here.
        # net.consolidation()

        print(modul_b_tmp.w.shape)

        # In Test Mode, network predicts labels for the unknown test data and the accuracy is calculated and stored.
        pred[edge_device_idx] = modul_b_tmp.test(np.array(feature_list_test_merged[edge_device_idx]))
        true_pos = np.sum(np.asarray(label_list_test_merged[edge_device_idx])[:, 0] == pred[edge_device_idx])
        accuracy[i-edge_device_idx*group_splitting, edge_device_idx] = true_pos / len(pred[edge_device_idx])
        print("Test Accuracy: {:.4f}".format(accuracy[i-edge_device_idx*group_splitting, edge_device_idx]))

        # When the Networks are trained (all groups shown) the different Networks are melded for a finale evaluation
        # Final, the results from the independent Devices for their independent Data as well as the results from the
        # melded Network is saved for later evaluation and visualization
        if i == (no_groups - 1):
            tmp_features = []
            tmp_labels = []
            modul_b[edge_device_idx] = modul_b_tmp

            tmp_modules = copy.deepcopy(modul_b)
            for j in range(len(modul_b)):
                for k in range(len(modul_b)):
                    if k != j:
                        print("Melding of Network {} with knowledge of Network {}".format(j, k))
                        modul_b[j].melding(tmp_modules[k])
                tmp_features = tmp_features + feature_list_test_merged[j]
                tmp_labels = tmp_labels + label_list_test_merged[j]

            feature_list_test_merged[-1] = tmp_features
            label_list_test_merged[-1] = tmp_labels

            pred[-1] = modul_b[-1].test(np.array(feature_list_test_merged[-1]))
            true_pos = np.sum(np.asarray(label_list_test_merged[-1])[:, 0] == pred[-1])
            print("Test Accuracy of melded Networks: {:.4f}".format(true_pos / len(pred[-1])))

    return label_list_test_merged, pred, accuracy
