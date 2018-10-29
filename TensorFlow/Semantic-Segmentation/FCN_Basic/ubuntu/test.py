'''
Created on Oct 29, 2018

@author: user170
'''

import tensorflow as tf

input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 3])
# static shape
print(input_tensor.shape)

conv_filter = tf.get_variable(
    'conv_filter', shape=[2, 2, 3, 6], dtype=tf.float32)
conv1 = tf.nn.conv2d(
    input_tensor, conv_filter, strides=[1, 2, 2, 1], padding='SAME')
# static shape
print(conv1.shape)

deconv_filter = tf.get_variable(
    'deconv_filter', shape=[2, 2, 6, 3], dtype=tf.float32)

deconv = tf.nn.conv2d_transpose(
    input_tensor,
    filter=deconv_filter,
    # use tf.shape to get the dynamic shape of the tensor
    # know at RUNTIME
    output_shape=tf.shape(input_tensor),
    strides=[1, 2, 2, 1],
    padding='SAME')
print(deconv.shape)