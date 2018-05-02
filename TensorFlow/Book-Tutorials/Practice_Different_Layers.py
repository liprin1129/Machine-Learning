import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
from tensorflow.python.framework import ops

'''
1. Convolutional Layer
2. Activation Layer
3. Max-Pool Layer
4. Fully Connected Layer
'''

#---------------------------------------------------|
#-------------------1D-data-------------------------|
#---------------------------------------------------|
ops.reset_default_graph()
sess = tf.Session()

# PARAMETER FOR THE RUN
data_size = 25
conv_size = 5
maxpool_size = 5
stride_size = 1

# FOR REPORUDUCIBILITY
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

# GENERATE 1D DATA
#data_1d = np.random.normal(size=data_size)
data_1d = np.random.randint(1, 11, size=data_size)
print(data_1d)
# PLACEHOLDER
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

#---------------------------
#--------Convolution--------
#---------------------------
## https://www.codeday.top/2017/10/10/48180.html

def conv_layer_1d(input_1d, filter, stride_1d):
    # TensorFlow's 'conv1d()' function only works with 3D arrays:
    # [batch#, width, channels(or dim)], we have 1 batch, and
    # width = 5, and 1 channel.
    # So next we create the 3D array by inserting dimension 1's.

    # [batch, length, dimension]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, -1)

    #print(np.shape(input_3d))

    convolution_output = tf.nn.conv1d(input_3d, filter, stride=stride_1d, padding='VALID')

    # GET RID OF EXTRA DIMENSIONS
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d

# CREATE FILTER FOR CONVOLUTION
## [filter width, input dimension, output dimension]
my_filter = tf.Variable(tf.ones([conv_size, 1, 1]))
#my_filter = tf.Variable(tf.truncated_normal([conv_size, 1, 1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter, stride_1d = stride_size)

#--------------------------
#--------Activation--------
#--------------------------
def activation(input_1d):
    return tf.nn.relu(input_1d)

# CREATE ACTIVATION LAYER
my_activation_output = activation(my_convolution_output)

#------------------------
#--------Max Pool--------
#------------------------
def max_pool(input_1d, width, stride):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, -1)
    
    pool_output = tf.layers.max_pooling1d(input_3d, width, stride, padding='VALID')

    # GET RID OF EXTRA DIMENSIONS
    pool_output_1d = tf.squeeze(pool_output)

    return pool_output_1d

my_maxpool_output = max_pool(my_activation_output, width=(maxpool_size), stride=(stride_size))

init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}

print('Input = array of length %d' % (x_input_1d.shape.as_list()[0]))
print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:' % 
      (conv_size,stride_size,my_convolution_output.shape.as_list()[0]))
print(np.shape(sess.run(my_convolution_output, feed_dict=feed_dict)))
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = above array of length %d' % (my_convolution_output.shape.as_list()[0]))
print('ReLU element wise returns an array of length %d:' % (my_activation_output.shape.as_list()[0]))
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = above array of length %d' % (my_activation_output.shape.as_list()[0]))
print('MaxPool, window length = %d, stride size = %d, results in the array of length %d' %
     (maxpool_size,stride_size,my_maxpool_output.shape.as_list()[0]))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
