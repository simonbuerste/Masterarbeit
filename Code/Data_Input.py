import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

"""
The Data Input is handled here. All necessary steps for preparing the Data for the Network is done
"""


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
    image = tf.cast(image, tf.float32)
    image = image/255

    img_shape = image.shape
    if img_shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    if img_shape[0] < 96:
        # Resize the image if required
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
    data_train = tfds.load(name=dataset, split=tfds.Split.TRAIN, as_supervised=True)
    data_test = tfds.load(name=dataset, split=tfds.Split.TEST, as_supervised=True)

    # For Visualization of before/after image with data augmentation
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

    data_train = data_train.repeat().shuffle(1024).batch(1)
    data_test = data_test.batch(1)

    return data_train, data_test
