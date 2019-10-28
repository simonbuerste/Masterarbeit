import tensorflow as tf

"""
Here the function for loading and returning of the pre-trained Module A is implemented.
"""


def modul_a(image_size=None):
    """
    Instantiates Modul A of the L DNN Algorithm
    # Arguments
        image_size: Size of the images which are read in
    # Returns
        modul_a:    Returns the built-up and pre-trained Modul A
    """
    if image_size is None:
        image_size = 224

    img_shape = (image_size, image_size, 3)

    # Create Modul A from the pre-trained Model MobileNet V2
    # Input Shape:  Shape of input data
    # Include Top:  If False, Network is used as Feature Extractor, Classification specific layers are removed
    # Weights:      Define on which data the model should have been pre-trained
    modul_a = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, pooling='max',
                                                weights='imagenet')

    # Freeze Modul A, because the parameters shouldn't be adjusted further
    modul_a.trainable = False

    # Write a summary of Modul A
    # modul_a.summary()

    return modul_a
