import tensorflow as tf
import tensorflow_datasets as tfds


def input_fn(dataset):
    mnist_train = tfds.load(name=dataset, split=tfds.Split.TRAIN)
    mnist_test = tfds.load(name=dataset, split=tfds.Split.TEST)

    mnist_test = tf.data.Dataset(mnist_test/255).astype(tf.float32)

    return mnist_train, mnist_test
