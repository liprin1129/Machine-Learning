import tensorflow as tf
from tensorflow.contrib.layers import flatten

import numpy as np

from data_wrangling import PickleHelper

net_setting = {"3x3_filter": lambda depth: tf.Variable(tf.truncated_normal((3, 3, 1, depth)))}

class VGG16(object):
    def __init__(self):
        self.__net_setting = net_setting

    def convolutional_node(self, in_data, name):
        with tf.variable_scope(name):
            conv_filter = self.filter_placeholder(name)
            conv_bias = self.
            conv = tf.nn.conv2d(in_data, conv1_filter, [1, 1, 1, 1], padding="SAME")
            
            
    def filter_constant(self, name):
        return tf.constant(self.__net_setting["3x3_filter"], name=name+"_filter")

    def bias_constant(self, name):
        return tf.constant(self.__net_setting
