import os
import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

"""
The Data Input is handled here. All necessary steps for preparing the Data for the Network is done
"""


def read_labels_mat(filename):
    """
    Reads the synsets for ImageNet from a .mat-File
    :param filename: Directory of the .mat-File
    :return: List of Synsets for mapping them to a label
    """
    labels = scipy.io.loadmat(filename)
    mapping = labels['synsets']
    synsets = []
    for idx, m in enumerate(mapping):
        synsets.append(mapping[idx][0][1][0])
    return synsets


def read_labels_txt(filename):
    """
    Reads a .txt-Label File
    :param filename: Directory of the .txt-Labelfile
    :return: The Labels as a numpy Array
    """
    file = open(filename, "r")
    labels = []
    for line in file:
        labels.append(line)
    return np.asarray(labels, dtype=int)


def _process_image(filename, label):
    """
    Reads the image from the file
    :param filename: Filename of the images
    :param label:    Label for each image
    :return: Image/Label Combination to store in a dataset
    """
    # Read the image file.
    height = 224
    width = 224
    img_raw = tf.io.read_file(filename)
    jpeg_img = tf.image.decode_jpeg(img_raw, channels=3)
    jpeg_img_resized = tf.image.resize(jpeg_img, (height, width))

    return jpeg_img_resized, label


def _process_dataset(all_train_img, all_train_label, all_test_img, all_test_label):
    """
    This Function processes an image dataset from local PC
    :param all_train_img:   Directory where all training images are stored
    :param all_train_label: File where the training labels (or mapping of folder name to label) are stored
    :param all_test_img:    Directory where all test images are stored
    :param all_test_label:  File where the test labels are stored
    :return: An training and test dataset
    """""
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

    return img_label_ds_train, img_label_ds_test


def extract_fn(tfrecord):
    """
    In this function datasets which are stored in a .tfrecord file are extracted and a dataset is returned which can be
    used further like any tf.dataset object
    # Arguments
        :param tfrecord:    TFRecord Dataset where all features are stored
    # Return
        image:              The extracted Image out of the TFRecord Dataset
        label:              The extracted label corresponding to the image
    """
    # Extract features using the keys set during creation
    features = {
        'image':    tf.io.FixedLenFeature([], tf.string),
        'label':    tf.io.FixedLenFeature([], tf.int64),
        'height':   tf.io.FixedLenFeature([], tf.int64),
        'width':    tf.io.FixedLenFeature([], tf.int64),
        'depth':    tf.io.FixedLenFeature([], tf.int64)
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)

    # Decode image and shape from tfrecord
    img_shape = tf.stack([sample['depth'], sample['height'], sample['width']])
    image = tf.io.decode_raw(sample['image'], tf.uint8)
    # reshape the image in "original Shape"
    image = tf.reshape(image, img_shape)
    # Transpose Image for tensorflow notation (heigth, width, num_channel)
    image = tf.transpose(image, (1, 2, 0))
    # Define new Size to resize the image
    image = tf.image.resize(image, (64, 64))

    label = sample['label']

    return [image, label]


def augmentation_fn(image, label):
    """
    The Image is augmented here. It is normalized to float-Values in Range [0, 1]. Additionally the size of the image
    and the number of channels is adjusted, e.g. Greyscale to RGB Conversion for MNIST Images
    # Arguments
        :param image: Image of the Dataset
        :param label: Label belonging to the Image
    # Return
        image: Returns the augmented image
        label: Returns the label
    """
    # Convert image to float and normalize it into [0, 1] Range
    image = tf.cast(image, tf.float32)
    image = image/255
    img_shape = image.shape

    # If just one channel is available, convert to RGB Image (Necessary for usage of pre-trained MobileNet
    if img_shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)

    # If Image Size is smaller than 96, resize the image (Necessary since smallest pre-trained Weights are trained for
    # 96x96x3 Images
    if img_shape[0] < 96:
        image = tf.image.resize(image, (96, 96))
    return image, label


def input_fn(dataset, visu):
    """
    The chosen dataset is (down-)loaded and split into Training and Test Data. If desired the Image can be shown before
    and after the Augmentation
    # Arguments
        :param dataset: String which gives the name of the Dataset which should be loaded
        :param visu: Bool which indicates if Example Image should be plotted
    # Return
        data_train: The Training Dataset containing all augmented Training Images and Labels
        data_test: The Test Dataset containing all augmented Test Images and Labels
    """
    try:
        data_train = tfds.load(name=dataset, split=tfds.Split.TRAIN, as_supervised=True)
        data_test = tfds.load(name=dataset, split=tfds.Split.TEST, as_supervised=True)
    except:
        dataset_folder = 'C:/Users/st158084/tensorflow_datasets'
        if dataset == 'imagenet10':
            filename_train = os.path.join(dataset_folder, dataset, 'train.tfrecords')
            filename_test = os.path.join(dataset_folder, dataset, 'test.tfrecords')

            data_train = tf.data.TFRecordDataset([filename_train])
            data_test = tf.data.TFRecordDataset([filename_test])
            data_train = data_train.map(extract_fn)
            data_test = data_test.map(extract_fn)
        elif dataset == 'imagenet':
            all_train_img = os.path.join(dataset_folder, dataset, 'img_train')
            all_train_label = os.path.join(dataset_folder, dataset, 'ILSVRC2012_devkit_t12/data/meta.mat')
            all_test_img = os.path.join(dataset_folder, dataset, 'img_val')
            all_test_label = os.path.join(dataset_folder, dataset, 'ILSVRC2012_validation_ground_truth.txt')

            data_train, data_test = _process_dataset(all_train_img, all_train_label, all_test_img, all_test_label)
        else:
            print('No such Dataset found')
            return

    # For Visualization of image before/after data augmentation
    if visu is True:
        example, = data_train.take(1)
        image, label = example[0], example[1]
        plt.figure(0)
        if image.shape[2] == 1:
            plt.imshow(tf.squeeze(image), cmap='gray')
        else:
            plt.imshow(image)
        print("Label: %d" % label.numpy())
        plt.show()

    data_train = data_train.map(augmentation_fn)
    data_test = data_test.map(augmentation_fn)

    # For Visualization of before/after image with data augmentation
    if visu is True:
        example, = data_train.take(1)
        image, label = example[0], example[1]
        plt.figure(1)
        plt.imshow(image)
        plt.show()
        print("Label: %d" % label.numpy())

    data_train = data_train.shuffle(1000, reshuffle_each_iteration=True).repeat().batch(1)
    data_test = data_test.batch(1)

    return data_train, data_test
