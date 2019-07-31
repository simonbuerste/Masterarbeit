import numpy as np
import random
import copy


def distributed_learning_fun(modul_b, no_classes, no_groups, feature_list_train, label_list_train, feature_list_test,
                             label_list_test):
    feature_list_test_merged = [[] for _ in range(2*len(modul_b))]
    label_list_test_merged = [[] for _ in range(2*len(modul_b))]
    pred = [[] for _ in range(2*len(modul_b))]

    classes = np.arange(0, 10)
    random.shuffle(classes)
    print(classes)

    group_splitting = np.ceil(no_groups/len(modul_b))

    edge_device_idx = 0

    for i in range(no_groups):  # - 1, -1, -1):
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
        for j in range(idx, (idx+group_size)):
            tmp_features = tmp_features + feature_list_train[classes[j]]
            tmp_labels = tmp_labels + label_list_train[classes[j]]

            feature_list_test_merged[edge_device_idx] = feature_list_test_merged[edge_device_idx] + feature_list_test[classes[j]]
            label_list_test_merged[edge_device_idx] = label_list_test_merged[edge_device_idx] + label_list_test[classes[j]]

        if i != 10:
            modul_b_tmp.train(np.array(tmp_features), np.asarray(tmp_labels)[:, 0], epochs=1)

        # net.consolidation()

        print(modul_b_tmp.w.shape)

        pred[edge_device_idx] = modul_b_tmp.test(np.array(feature_list_test_merged[edge_device_idx]))
        true_pos = np.sum(np.asarray(label_list_test_merged[edge_device_idx])[:, 0] == pred[edge_device_idx])
        print("Test Accuracy: {:.4f}".format(true_pos / len(pred[edge_device_idx])))

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

            for j in range(len(modul_b), 2*len(modul_b)):
                feature_list_test_merged[j] = tmp_features
                label_list_test_merged[j] = tmp_labels

                pred[j] = modul_b[j-len(modul_b)].test(np.array(feature_list_test_merged[j]))
                true_pos = np.sum(np.asarray(label_list_test_merged[j])[:, 0] == pred[j])
                print("Test Accuracy of melded Networks: {:.4f}".format(true_pos / len(pred[j])))

    return label_list_test_merged, pred
