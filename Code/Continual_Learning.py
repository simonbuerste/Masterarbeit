import numpy as np
import random

"""
Function for the continual incremental class learning is implemented here
"""


def continual_learning_fun(modul_b, no_classes, no_groups, feature_list_train, label_list_train, feature_list_test,
                           label_list_test):
    """
    Incremental Class Learning is executed here. The classes are learned one after each other and testing happens
    on all previously learned classes
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
    # for i in range(no_classes):
    #     if i == 0:
    #         feature_list_test_merged = feature_list_test[i]
    #         label_list_test_merged = label_list_test[i]
    #     else:
    #         feature_list_test_merged = feature_list_test_merged + feature_list_test[i]
    #         label_list_test_merged = label_list_test_merged + label_list_test[i]

    # Create Lists for Test Data and Accuracy
    feature_list_test_merged = []
    label_list_test_merged = []
    accuracy = np.zeros((no_groups, 1))

    # Random shuffle the classes for reliable Results with more repetitions
    classes = np.arange(0, no_classes)
    random.shuffle(classes)
    print(classes)

    for i in range(no_groups):  # - 1, -1, -1):
        print("------ Training Group {} of {} ------".format(int(i+1), no_groups))

        group_size = int(no_classes/no_groups)
        idx = i*group_size

        tmp_features = []
        tmp_labels = []
        # Get all training and test data from the Group. A group may contain more than one class (defined by group_size)
        for j in range(idx, (idx+group_size)):
            tmp_features = tmp_features + feature_list_train[classes[j]]
            tmp_labels = tmp_labels + label_list_train[classes[j]]

            feature_list_test_merged = feature_list_test_merged + feature_list_test[classes[j]]
            label_list_test_merged = label_list_test_merged + label_list_test[classes[j]]

        # A Group can be selected which should not be trained
        if i != 10:
            modul_b.train(np.array(tmp_features), np.asarray(tmp_labels)[:, 0], epochs=1)

        # If desired, Consolidation of Weights for all classes can be done here.
        # net.consolidation()

        print(modul_b.w.shape)

        # In Test Mode, network predicts labels for the unknown test data and the accuracy is calculated and stored.
        pred = modul_b.test(np.array(feature_list_test_merged))
        true_pos = np.sum(np.asarray(label_list_test_merged)[:, 0] == pred)
        accuracy[i] = true_pos / len(pred)
        print("Test Accuracy: {}".format(accuracy[i]))

    # Store the results in a nested list for correct handling in the evaluation/plotting functions
    tmp_list_1 = [[] for _ in range(1)]
    tmp_list_2 = [[] for _ in range(1)]
    tmp_list_1[0] = label_list_test_merged
    tmp_list_2[0] = pred
    return tmp_list_1, tmp_list_2, accuracy
