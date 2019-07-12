import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


def augmentation_fn(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255

    img_shape = image.shape
    if img_shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    if img_shape[0] < 96:
        # Resize the image if required
        image = tf.image.resize(image, (96, 96))
    return image, label


def input_fn(dataset):
    data_train = tfds.load(name=dataset, split=tfds.Split.TRAIN, as_supervised=True)
    data_test = tfds.load(name=dataset, split=tfds.Split.TEST, as_supervised=True)

    # For Visualization of before/after image with data augmentation
    # example, = data_train.take(1)
    # image, label = example[0], example[1]
    # plt.figure(0)
    # plt.imshow(image)  # , cmap=plt.get_cmap("gray"))
    # print("Label: %d" % label.numpy())

    data_train = data_train.map(augmentation_fn)
    data_test = data_test.map(augmentation_fn)

    # For Visualization of before/after image with data augmentation
    # example, = data_train.take(1)
    # image, label = example[0], example[1]
    # plt.figure(1)
    # plt.imshow(image)
    # print("Label: %d" % label.numpy())

    data_train = data_train.repeat().shuffle(1024).batch(1)
    data_test = data_test.batch(1)

    return data_train, data_test
