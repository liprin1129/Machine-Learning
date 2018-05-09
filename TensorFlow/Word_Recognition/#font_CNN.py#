#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

global_path = os.path.abspath("/mnt/SharedData/Development/Personal_Dev/Machine-Learning/One_Layer/")
global_path = os.path.abspath("../One_Layer/")

if global_path not in sys.path:
    sys.path.append(global_path)

from pickle_helper import PickleHelper

import numpy as np
from tqdm import tqdm

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import flatten
#from tensorflow.python.framework import ops

# START A GRAPH SESSION
#sess = tf.Session()
#sess = tf.InteractiveSession()

# --------------------- #
# ----- LOAD DATA ----- #
# --------------------- #
'''
font_data = PickleHelper.load_pickle(
        path="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/",
        name = 'mnist.pkl')
'''
data_dir = '../Data/Font data/'
font_folders = os.listdir(data_dir)

font_imgs_cv2 = []
font_labels = []

test_font_imgs_cv2 = []
test_font_labels = []

#def load_img_and_set_labels(data):
for idx, folder in enumerate(font_folders):
    font_folder = data_dir+folder+'/'
    font_img_lists = os.listdir(font_folder)

    print(font_folder)
    if idx == 0:
        for font_img_name in os.listdir(font_folder):
            font_imgs_cv2.append(cv2.imread(font_folder+font_img_name, cv2.IMREAD_UNCHANGED))
            font_labels.append(font_img_name[:-4])

    elif idx == 1:
        for font_img_name in os.listdir(font_folder):
            test_font_imgs_cv2.append(cv2.imread(font_folder+font_img_name, cv2.IMREAD_UNCHANGED))
            test_font_labels.append(font_img_name[:-4])

#print(font_labels[30], test_font_labels[30])
'''
import time
fig, ax = plt.subplots(1, 2, figsize=(5, 5))
for _ in range(10):
    idx = np.random.randint(2136)
    ax[0].imshow(font_imgs_cv2[idx], cmap='gray')
    ax[0].set_title(font_labels[idx])
    
    ax[1].imshow(test_font_imgs_cv2[idx], cmap='gray')
    ax[1].set_title(test_font_labels[idx])
    plt.draw()
    plt.pause(1)
'''

# - Normalisation
def normalize_cols(x, y):
    train_x = np.array(x, dtype=np.float32)
    train_y = np.array(y, dtype=np.float32)
    
    train_x = np.reshape(train_x, (-1, 784))
    col_max = np.max(train_x, axis=1)
    col_min = np.min(train_x, axis=1)
    min_max_norm = np.divide(
        (np.subtract(train_x.T, col_min)),
        (np.subtract(col_max, col_min)))
    return np.reshape(np.transpose(min_max_norm), (-1, 28, 28)), train_y
 
train_xdata, train_labels = normalize_cols(font_imgs_cv2, font_labels)
test_xdata, test_labels = normalize_cols(test_font_imgs_cv2, test_font_labels)

print('\n<--- TRAIN DATA --->')
print('DATA SHAPES: ', train_xdata.shape, train_labels.shape)
print('DATA TYPES: ', type(train_xdata[0, 0, 0]), type(train_labels[0]))
print('DATA MIN MAX: ', np.min(train_xdata[0]), np.max(train_xdata[0]))

print('<--- TEST DATA --->')
print('DATA SHAPES: ', test_xdata.shape, test_labels.shape)
print('DATA TYPES: ', type(test_xdata[0, 0, 0]), type(test_labels[0]))
print('DATA MIN MAX: ', np.min(test_xdata[0]), np.max(test_xdata[0]), '\n')

# --------------------- #
# ----- VARIABLES ----- #
# --------------------- #

# MODEL PARAMETERS
img_channel = 1
learning_rate = 0.001
epoch = 20000
batch_size = 2136
#drop_prop = 0.4

# PLACEHOLDERS
img_shape = (None, 28, 28, img_channel)
x = tf.placeholder(tf.float32, shape=img_shape)

labels = tf.placeholder(tf.int32, shape=(None))
y_ = tf.one_hot(labels, depth=len(train_labels), dtype=tf.float32)

keep_prob = tf.placeholder(tf.float32)
'''
with tf.Session() as sess:
    a = sess.run(y_, feed_dict={labels:train_labels})
    print(np.shape(a))
''' 

eval_in_layer = tf.placeholder(tf.float32, shape=img_shape)
eval_out_layer = tf.placeholder(tf.int32, shape=(None))

# CONVOLUTIONAL 1
conv1_kernel = 5
conv1_filter = 32
conv1_stride = 1

pool1_kernel = 2
pool1_stride = 2

conv1_W = tf.Variable(
    tf.truncated_normal(
        [conv1_kernel, conv1_kernel, img_channel, conv1_filter],
        stddev=0.1, dtype=tf.float32)) # 5x5 filter shape, 1->32

conv1_b = tf.Variable(
    tf.zeros([conv1_filter], dtype=tf.float32))

# CONVOLUTIONAL 2

conv2_kernel = 5
conv2_filter = 64
conv2_stride = 1

pool2_kernel = 2
pool2_stride = 2

conv2_W = tf.Variable(
    tf.truncated_normal(
        [conv2_kernel, conv2_kernel, conv1_filter, conv2_filter],
        stddev=0.1, dtype=tf.float32)) # 5x5 filter shape, 32->64

conv2_b = tf.Variable(
    tf.zeros([conv2_filter], dtype=tf.float32))

# DENSE 1
dense1_in = 3136
dense1_out = 3000
drop1_prop = 0.1

dense1_W = tf.Variable(
    tf.truncated_normal(
        [dense1_in, dense1_out], stddev=0.1, dtype=tf.float32))

dense1_b = tf.Variable(
    tf.zeros([dense1_out], dtype=tf.float32))


# DENSE 2
dense2_out = 2136

dense2_W = tf.Variable(
    tf.truncated_normal(
        [dense1_out, dense2_out], stddev=0.1, dtype=tf.float32))

dense2_b = tf.Variable(
    tf.zeros([dense2_out], dtype=tf.float32))

# ----------------- #
# ----- Model ----- #
# ----------------- #

def cnn_dense_model(features):
    #input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # CONV1 -> RELU1 -> MAXPOOL1
    conv1 = tf.nn.conv2d(features,
                         conv1_W,
                         strides=[conv1_stride, conv1_stride, conv1_stride, conv1_stride],
                         padding='SAME')

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    max_pool1 = tf.nn.max_pool(relu1,
                               ksize=[1, pool1_kernel, pool1_kernel, 1],
                               strides=[1, pool1_stride, pool1_stride, 1],
                               padding='VALID')

    # CONV2 -> RELU2 -> MAXPOOL2
    conv2 = tf.nn.conv2d(max_pool1,
                         conv2_W,
                         strides=[conv2_stride, conv2_stride, conv2_stride, conv2_stride],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    max_pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, pool2_kernel, pool2_kernel, 1],
                               strides=[1, pool2_stride, pool2_stride, 1],
                               padding='VALID')

    # DENS1 -> DROPOUT
    flatten_node = flatten(max_pool2)
    
    dense1 = tf.nn.relu(tf.add(tf.matmul(flatten_node, dense1_W), dense1_b))
    dropout1 = tf.nn.dropout(dense1, keep_prob)

    # DENS2 -> LOGITS
    logits = tf.nn.relu(tf.add(tf.matmul(dropout1, dense2_W), dense2_b))

    return logits

# CALCULATE LOSS (for both TRAIN and EVAL modes)
logits = cnn_dense_model(x)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss=cross_entropy)                                

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    one_hot_train_labels = sess.run(y_, feed_dict={labels:train_labels})
    one_hot_test_labels = sess.run(y_, feed_dict={labels:test_labels})

    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
    
    for i in tqdm(range(epoch)):
        batch_index = np.random.choice(len(train_xdata), size=batch_size)
        batch_x = np.expand_dims(train_xdata[batch_index], axis=-1)
        batch_y = one_hot_train_labels[batch_index]

        #train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: drop_prop})
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: drop1_prop})

        if i % 100 == 0:
            #train_accuracy = accuracy.eval(feed_dict={
            #    x: batch_x, y_: batch_y, keep_prob: 1.0})
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch_x, y_: batch_y, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

            test_batch_x = np.expand_dims(test_xdata[batch_index], axis=-1)
            test_batch_y = one_hot_test_labels[batch_index]

            print('Test accuracy %g' % accuracy.eval(feed_dict={
                x: test_batch_x, y_: test_batch_y, keep_prob: 1.0}))
            
            train_pred = sess.run(logits, feed_dict={x: batch_x,
                                                     y_: batch_y,
                                                     keep_prob: 1})
            test_pred = sess.run(logits, feed_dict={x: test_batch_x,
                                                     y_: test_batch_y,
                                                     keep_prob: 1})

            ax[0, 0].set_title('Train True')
            ax[0, 0].imshow(train_xdata[np.argmax(batch_y[0])], cmap='gray')
            ax[0, 1].set_title('Train Predict')
            ax[0, 1].imshow(train_xdata[np.argmax(train_pred[0])], cmap='gray')

            ax[1, 0].set_title('Test True')
            ax[1, 0].imshow(test_xdata[np.argmax(test_batch_y[0])], cmap='gray')
            ax[1, 1].set_title('Test Predict')
            ax[1, 1].imshow(test_xdata[np.argmax(test_pred[0])], cmap='gray')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            #b = sess.run(logits, feed_dict={x: batch_x, y_: batch_y, keep_prob: drop1_prop})
