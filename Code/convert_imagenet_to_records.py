from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

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


def read_labels_txt(filename):
    file = open(filename, "r")
    labels = []
    for line in file:
        labels.append(line)
    return np.asarray(labels, dtype=int)


def encode_jpeg(image):
    jpeg_image = tf.image.decode_jpeg(image, channels=3)
    return jpeg_image


def _process_image(filename):
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
    image = tf.io.read_file(filename)
    jpeg_data = encode_jpeg(image)
    jpeg_data_resized = tf.image.resize(jpeg_data, (height, width))

    return jpeg_data_resized, height, width


def _process_dataset(input_dir, label_file, tfrecord_file):
    """Process a complete data set and save it as a TFRecord.
    """
    files = tf.io.gfile.listdir(input_dir)
    label = read_labels_txt(label_file) - 1
    i = 0

    pb = ProgressBar(total=100)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for f in files:
            if f.endswith('JPEG'):
                filename = os.path.join(input_dir, f)
                img_data, height, width = _process_image(filename)

                # plt.figure(0)
                # plt.imshow(np.asarray(img_data)/255)

                example = _convert_to_example(np.asarray(img_data), label[i], height, width)
                writer.write(example.SerializeToString())
                i += 1
                pb.print_progress_bar(i/len(label))


if __name__ == '__main__':
    data_dir = 'C:/Users/st158084/tensorflow_datasets/imagenet2012/ILSVRC2012_img_val'
    mode = 'test'
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    label_filename = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')

    _process_dataset(data_dir, label_filename, output_file)
