"""
This file is the Main File for executing the Lifelong DNN Algorithm
Pipeline is set up here and all relevant Parameters can be set here
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from data_input import input_fn
from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP

no_classes = 10
no_groups = 10
train_img_per_class = 250
test_img_per_class = 1000

dataset = "mnist"
data_train, data_test = input_fn(dataset)

modulA = modul_a(image_size=96)

feature_list_train = [[] for j in range(no_classes)]
feature_list_test = [[] for j in range(no_classes)]
label_list_train = [[] for j in range(no_classes)]
label_list_test = [[] for j in range(no_classes)]
i = np.zeros((no_classes,), dtype=int)

for img, label in data_train:

    all_data = i >= train_img_per_class
    if np.all(all_data == True):
        break

    label_scalar = (np.array(label)).item()
    if i[label_scalar] < train_img_per_class:
        features = modulA.predict(img, steps=1)
        features_np = np.array(features)
        feature_list_train[label_scalar].append(features_np.flatten())
        label_list_train[label_scalar].append(label)
        i[label_scalar] += 1

i = np.zeros((no_classes,), dtype=int)
for img, label in data_test:

    all_data = i >= test_img_per_class
    if np.all(all_data == True):
        break

    label_scalar = (np.array(label)).item()
    if i[label_scalar] < test_img_per_class:
        features = modulA.predict(img, steps=1)
        features_np = np.array(features)
        feature_list_test[label_scalar].append(features_np.flatten())
        label_list_test[label_scalar].append(label)
        i[label_scalar] += 1


no_epoch = 10
net = FuzzyARTMAP(alpha=0.25, rho=0.65, n_classes=no_classes, s=1.05)

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
        net.train(np.array(feature_list_train[i]), np.asarray(label_list_train[i])[:, 0], epochs=1)

    # net.consolidation()

    print(net.w.shape)

    if i == 0:
        feature_list_test_merged = feature_list_test[i]
        label_list_test_merged = label_list_test[i]
    else:
        feature_list_test_merged = feature_list_test_merged + feature_list_test[i]
        label_list_test_merged = label_list_test_merged + label_list_test[i]

    if ((i+1) % (no_classes/no_groups)) == 0:
        pred = net.test(np.array(feature_list_test_merged))
        true_pos = np.sum(np.asarray(label_list_test_merged)[:, 0] == pred)
        print("Test Accuracy: {:.4f}".format(true_pos/len(pred)))

cm = confusion_matrix(y_true=np.asarray(label_list_test_merged), y_pred=pred)
cm = cm/cm.sum(axis=1)[:, None]

fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       title='Confusion Matrix of L DNN Algorithm')
# plt.matshow(cm)
# plt.title('Confusion Matrix of Classifier')
# plt.colorbar()

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.mean()
for k in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, k, format(cm[k, j], fmt),
                ha="center", va="center",
                color="white")

plt.show()
