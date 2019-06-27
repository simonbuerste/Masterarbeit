"""
This file is the Main File for executing the Lifelong DNN Algorithm
Pipeline is set up here and all relevant Parameters can be set here
"""

import tensorflow as tf

from data_input import input_fn
from ModulA import modul_a

dataset = "mnist"

mnist_train, mnist_test = input_fn(dataset)
