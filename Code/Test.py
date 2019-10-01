from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import pathlib
import cv2
import scipy.io

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from console_progressbar import ProgressBar


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    height: integer, image height in pixels
    width: integer, image width in pixels
    Returns:
    Example proto
    """

    channels = 3

    example = tf.train.Example(features=tf.train.Features(feature={
        'height':   _int64_feature(height),
        'width':    _int64_feature(width),
        'depth':    _int64_feature(channels),
        'label':    _int64_feature(label),
        'image':    _bytes_feature(image_buffer.tostring())}))
    return example


def read_labels_mat(filename):
    labels = scipy.io.loadmat(filename)
    mapping = labels['synsets']
    idx = np.argwhere('n01440764' == mapping)
    synsets = []
    for idx, m in enumerate(mapping):
        synsets.append(mapping[idx][0][1][0])
    return synsets


def read_labels_txt(filename):
    file = open(filename, "r")
    labels = []
    for line in file:
        labels.append(line)
    return np.asarray(labels, dtype=int)


def _process_image(filename, label):
    """Process a single image file.
    Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    height = 224
    width = 224
    img_raw = tf.io.read_file(filename)
    jpeg_img = tf.image.decode_jpeg(img_raw, channels=3)
    jpeg_img_resized = tf.image.resize(jpeg_img, (height, width))

    return jpeg_img_resized, label


def _process_dataset(all_train_img, all_train_label, all_test_img, all_test_label):
    """Process a complete data set and save it as a TFRecord.
    """
    # Read all training and test images and set the correct path
    train_files = tf.io.gfile.listdir(all_train_img)
    test_files = tf.io.gfile.listdir(all_test_img)
    all_train_class_path = [os.path.join(all_train_img, f) for f in train_files]
    all_test_img_path = [os.path.join(all_test_img, f) for f in test_files]
    # Since Labels start at 1, substract -1 for correct indices with starting '0'
    label_np_test = read_labels_txt(all_test_label) - 1
    synsets_np_train = read_labels_mat(all_train_label)

    all_train_img_path = []
    label_np_train = []
    for folder in all_train_class_path:
        img_class_files = tf.io.gfile.listdir(folder)
        synset = os.path.basename(os.path.normpath(folder))
        label_train = synsets_np_train.index(synset)
        for f in img_class_files:
            all_train_img_path.append(os.path.join(folder, f))
            label_np_train.append(label_train)

    # Create the Datasets for training and test images with corresponding labels
    path_ds_train = tf.data.Dataset.from_tensor_slices((all_train_img_path, label_np_train))
    img_label_ds_train = path_ds_train.map(_process_image)
    path_ds_test = tf.data.Dataset.from_tensor_slices((all_test_img_path, label_np_test))
    img_label_ds_test = path_ds_test.map(_process_image)

    print(img_label_ds_train)
    print(img_label_ds_test)

    # Check an example image if necessary
    # example, = img_label_ds_test.take(1)
    for i in range(5):
        example, = img_label_ds_train.take(1)
        image, label = example[0], example[1]
        plt.figure(i)
        if image.shape[2] == 1:
            plt.imshow(tf.squeeze(image), cmap='gray')
        else:
            plt.imshow(image/255)
        print("Label: {}".format(label.numpy()))
        plt.show()

    return img_label_ds_train, img_label_ds_test


if __name__ == '__main__':
    data_dir = 'C:/Users/st158084/tensorflow_datasets/imagenet2012'

    train_img_dir = os.path.join(data_dir, 'img_train')
    label_file_train = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat')

    test_img_dir = os.path.join(data_dir, 'img_val')
    label_file_test = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')

    data_train, data_test = _process_dataset(train_img_dir, label_file_train, test_img_dir, label_file_test)
