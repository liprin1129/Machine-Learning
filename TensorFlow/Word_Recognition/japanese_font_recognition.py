import os
import sys

global_path = os.path.abspath("/mnt/SharedData/Development/Personal_Dev/Machine-Learning/One_Layer/")

if global_path not in sys.path:
    sys.path.append(global_path)

from pickle_helper import PickleHelper

import numpy as np
#import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import flatten
from tensorflow.python.framework import ops

from font_CNN import font_recognition
fonts_pkl = os.listdir("/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/")

# START A GRAPH SESSION
sess = tf.Session()

'''
# -------------------------
# CHECK DATA CORRESPONDENCE
# -------------------------
font_data1 = PickleHelper.load_pickle(
    path="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/",
    name = fonts_pkl[0])
font_data2 = PickleHelper.load_pickle(
    path="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/",
    name = fonts_pkl[5])

train_xdata1 = font_data1[0]
train_ydata1 = [i for i in range(0, len(train_xdata1))]

train_xdata2 = font_data2[0]
train_ydata2 = [i for i in range(0, len(train_xdata2))]

print(np.shape(train_xdata1[0]))
#np.reshape(train_xdata1[0], (28, 28))

import time

for i in range(10):

    index = np.random.randint(len(train_xdata1))
    font_img1 = np.reshape(train_xdata1[index], (28, 28))
    font_img2 = np.reshape(train_xdata2[index], (28, 28))

    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].imshow(font_img1, cmap='gray')
    ax[1].imshow(font_img2, cmap='gray')
    plt.show()
# -------------------------
'''

def font_load(idx):
    font_data = PickleHelper.load_pickle(
        path="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/",
        name = fonts_pkl[idx])

    train_xdata = np.array(font_data[0])
    #train_xdata = np.reshape(font_data[0], (-1, 28, 28))
    train_ydata = np.array([i for i in range(0, len(train_xdata))])

    return train_xdata, train_ydata

'''
### NORMALIZATION BY COLUMN
def normalize_cols(m):
    m = np.array(m)
    col_max = np.max(m, axis=0)
    col_min = np.min(m, axis=0)
    return (m-col_min) / (col_max - col_min)
'''

train_xdata, train_ydata = font_load(0)
# NOMALIZATION
train_xdata = train_xdata/255
train_xdata = np.reshape(train_xdata, (-1, 28, 28))
train_xdata = np.expand_dims(train_xdata, -1)

#train_ydata = tf.one_hot(train_ydata, 2136)
train_ydata = train_ydata/2135

eval_index = np.random.choice(len(train_xdata), size=1)
print('TRAIN SHAPE: ', np.shape(train_xdata))#, train_xdata[eval_index[0]])
print('LABEL SHAPE: ', np.shape(train_ydata), train_ydata[0])#, train_ydata[eval_index[0]])
'''
font_recognition(train_xdata, train_ydata, None, None, learning_rate = 0.1,
                 conv1_filter=2, conv1_depth = 10, max_pool_size1 = 2,
                 conv2_filter=4, conv2_depth = 60, max_pool_size2 = 4,
                 fully_size1= 1800, target_size = 2136)
'''

image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
num_channels = 1
conv1_depth = 25
conv2_depth = 50
fully_connected_size1 = 100
output_size = 1
max_pool_size1 = 2
max_pool_size2 = 2

# MODEL PLACEHOLDER
x_input_shape = (None, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.float32, shape=(None))

eval_input_shape = (None, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape=(None))

# CONVOLUTIONAL LAYER VARIABLES
conv1_W = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_depth],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv1_b = tf.Variable(tf.zeros([conv1_depth], dtype=tf.float32))

conv2_W = tf.Variable(tf.truncated_normal([4, 4, conv1_depth, conv2_depth],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv2_b = tf.Variable(tf.zeros([conv2_depth], dtype=tf.float32))

fully1_W = tf.Variable(tf.truncated_normal([4*4*50, fully_connected_size1], stddev=0.1, dtype=tf.float32))
fully1_b = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

fully2_W = tf.Variable(tf.truncated_normal([fully_connected_size1, output_size], stddev=0.1, dtype=tf.float32))
fully2_b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1, dtype=tf.float32))

def my_conv_net(input_data):
    # FIRST CONV -> RELU -> MAXPOOL
    conv1 = tf.nn.conv2d(input_data, conv1_W, strides=[1,1,1,1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='VALID')

    # FIRST CONV -> RELU -> MAXPOOL
    conv2 = tf.nn.conv2d(max_pool1, conv2_W, strides=[1,1,1,1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides = [1, max_pool_size2, max_pool_size2, 1], padding='VALID')

    # FLATTEN
    flatten_node = flatten(max_pool2)


    # FIRST FULLY
    fully_connected_1 = tf.nn.relu(tf.add(tf.matmul(flatten_node, fully1_W), fully1_b))

    # SECOND FULLY
    final_model_output = tf.add(tf.matmul(fully_connected_1, fully2_W), fully2_b)

    return(final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

# PREDICTION
#prediction = my_conv_net(x_input)

# LOSS FUNCTION (softmax cross entropy)
loss = tf.reduce_mean(tf.square(y_target - model_output))

# OPTIMIZER
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# INITIALIZE
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
test_loss = []

#train_xdata = np.expand_dims(train_xdata[0], 0).astype(np.float32)
#sess.run(train_step, feed_dict={x_input:train_xdata, y_target:train_ydata[0]})
#sess.run(model_output, feed_dict={x_input:train_xdata})
#print(type(train_xdata[0, 0, 0, 0]))

for i in range(100):
    feed_x = np.expand_dims(train_xdata[0], 0).astype(np.float32)
    feed_y = train_ydata[0].astype(np.float32)

    prediction = sess.run(model_output, feed_dict={x_input:feed_x})
    sess.run(train_step, feed_dict={x_input:feed_x, y_target:feed_y})
    
    temp_loss = sess.run(loss, feed_dict={x_input:feed_x, y_target:feed_y})

    if (i+1) % 5:
        #print(prediction, feed_y, temp_loss)
        #fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        
