"""
This file is the Main File for executing the Lifelong DNN Algorithm
Pipeline is set up here and all relevant Parameters can be set here
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

from data_input import input_fn
from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP

dataset = "mnist"
data_train, data_test = input_fn(dataset)

modulA = modul_a(image_size=96)
#
# model = tf.keras.Sequential([modulA,
#                              keras.layers.GlobalAveragePooling2D(),
#                              keras.layers.Dense(512, activation='relu'),
#                              keras.layers.Dense(10, activation='softmax')])
#
# model.summary()
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy', 'sparse_categorical_accuracy'])
#
# epochs = 5
no_classes = 10
train_img_per_class = 10
test_img_per_class = 10
#
# model.fit(data_train, epochs=epochs, steps_per_epoch=steps_per_epoch_train)
# model.evaluate(data_test, steps=steps_per_epoch_test)

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

    all_data = i >= train_img_per_class
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
net = FuzzyARTMAP(alpha=0.25, rho=0.55, n_classes=no_classes)
for i in range(no_classes):
    print("------ Training Class {} ------".format(i))
    net.train(np.array(feature_list_train[i]), np.asarray(label_list_train[i])[:, 0], epochs=1)
    print(net.w.shape)

    if i == 0:
        feature_list_test_merged = feature_list_test[i]
        label_list_test_merged = label_list_test[i]
    else:
        feature_list_test_merged = feature_list_test_merged + feature_list_test[i]
        label_list_test_merged = label_list_test_merged + label_list_test[i]

    pred = net.test(np.array(feature_list_test_merged))
    true_pos = np.sum(np.asarray(label_list_test_merged)[:, 0] == pred)
    print("Test Accuracy: {:.4f}".format(true_pos/len(pred)))
