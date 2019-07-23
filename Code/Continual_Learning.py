import numpy as np


def continual_learning_fun(modul_b, no_classes, no_groups, feature_list_train, label_list_train, feature_list_test,
                           label_list_test):
    """
    Incremental Class Learning is executed here. The classes are learned one after each other and testing happens
    on all previously learned classes
    # Arguments
        :param modul_b: The created Module B which is in the L DNN Algorithm the incremental Classifier. Class Object
                        with defined Functions and Parameters
        :param no_classes: Number of classes in the dataset
        :param no_groups: Number of Training Groups. Can be used if more classes should be trained together
        :param feature_list_train: The extracted Features from training data separated by each class
        :param label_list_train: The corresponding Labels to the training data
        :param feature_list_test: The extracted Features from test data separated by each class
        :param label_list_test: The corresponding Labels to the training data
    # Returns
        label_list_test_merged: The merged List of Labels for the test data
        pred: Predicted Labels from the Network --> Output of the L DNN Algorithm
    """
    # for i in range(no_classes):
    #     if i == 0:
    #         feature_list_test_merged = feature_list_test[i]
    #         label_list_test_merged = label_list_test[i]
    #     else:
    #         feature_list_test_merged = feature_list_test_merged + feature_list_test[i]
    #         label_list_test_merged = label_list_test_merged + label_list_test[i]

    for i in range(no_classes):
        print("------ Training Class {}------".format(int(i)))
        # idx = int(i*no_classes/no_groups)
        # tmp_features = feature_list_train[idx] + feature_list_train[idx+1]
        # tmp_labels = label_list_train[idx] + label_list_train[idx+1]
        # net.train(np.array(tmp_features), np.asarray(tmp_labels)[:, 0], epochs=1)

        if i != 10:
            modul_b.train(np.array(feature_list_train[i]), np.asarray(label_list_train[i])[:, 0], epochs=1)

        # net.consolidation()

        print(modul_b.w.shape)

        if i == 0:
            feature_list_test_merged = feature_list_test[i]
            label_list_test_merged = label_list_test[i]
        else:
            feature_list_test_merged = feature_list_test_merged + feature_list_test[i]
            label_list_test_merged = label_list_test_merged + label_list_test[i]

        if ((i + 1) % (no_classes / no_groups)) == 0:
            pred = modul_b.test(np.array(feature_list_test_merged))
            true_pos = np.sum(np.asarray(label_list_test_merged)[:, 0] == pred)
            print("Test Accuracy: {:.4f}".format(true_pos / len(pred)))

    return label_list_test_merged, pred
