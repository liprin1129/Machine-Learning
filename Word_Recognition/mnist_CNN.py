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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import flatten
#from tensorflow.python.framework import ops

# START A GRAPH SESSION
#sess = tf.Session()
#sess = tf.InteractiveSession()
# DOWNLOAD DATA
'''
font_data = PickleHelper.load_pickle(
        path="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Font data/",
        name = 'mnist.pkl')
'''
data_dir = '../Data/HandWritten_Data/'
mnist = read_data_sets(data_dir)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

train_labels = mnist.train.labels
test_labels = mnist.test.labels

#print(np.shape(train_xdata[0]))

# --------------------- #
# ----- VARIABLES ----- #
# --------------------- #

# MODEL PARAMETERS
img_channel = 1
learning_rate = 0.001
epoch = 20000
batch_size = 100
drop_prop = 0.4

# PLACEHOLDERS
img_shape = (None, 28, 28, img_channel)
x = tf.placeholder(tf.float32, shape=img_shape)

labels = tf.placeholder(tf.int32, shape=(None))
y_ = tf.one_hot(labels, depth=10, dtype=tf.float32)

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
dense1_out = 1024
drop_prop = 0.4

dense1_W = tf.Variable(
    tf.truncated_normal(
        [dense1_in, dense1_out], stddev=0.1, dtype=tf.float32))

dense1_b = tf.Variable(
    tf.zeros([dense1_out], dtype=tf.float32))

# DENSE 2
dense2_out = 10

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
    
    for i in range(epoch):
        batch_index = np.random.choice(len(train_xdata), size=batch_size)
        batch_x = np.expand_dims(train_xdata[batch_index], axis=-1)
        batch_y = one_hot_train_labels[batch_index]

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_x, y_: batch_y, keep_prob: 1.0})
            #train_accuracy = sess.run(accuracy, feed_dict={
            #    x: batch_x, y_: batch_y, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: drop_prop})
        #sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: drop_prop})

        #print('test accuracy %g' % accuracy.eval(feed_dict={
        #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

"""
def cnn_dense_evaluate(logits, labels):
        pred = np.argmax(logits, axis=1)
        #np.argmax(tf.nn.softmax(logits, name='softmax_tensor'), axis=1)
        num_correct = np.sum(np.equal(pred, labels))
        return(100. * num_correct/pred.shape[0])
        

# INITIALIZE
sess.run(tf.global_variables_initializer())

for i in tqdm(range(epoch)):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    batch_x = np.expand_dims(train_xdata[rand_index], axis=-1)
    batch_y = train_labels[rand_index]

    train_dict = {in_true: batch_x}

    #train_stage = cnn_dense_model(in_true)
    out_logits = sess.run(logits, feed_dict=train_dict)

    '''
    if (i+1) % 10 == 0:
        test = cnn_dense_evaluate(out_logits, batch_y)
        print(test)
    '''
        #print('SHAPE: ', np.shape(test), 'ARGMAX: ', np.argmax(test, axis=1))
        
    '''
    if (i+1) % 10 == 0:
        rand_index = np.random.choice(len(test_xdata), size=batch_size)
        batch_x = np.expand_dims(test_xdata[rand_index], axis=-1)
        batch_y = test_labels[rand_index]
        
        test_dict = {in_true: batch_x, out_true: batch_y}
        evaluate_stage = cnn_dense_train(in_true, out_true)
        print(sess.run(evaluate_stage, train_dict))
    '''

'''
train_xdata = np.expand_dims(train_xdata[10], -1)
train_xdata = np.expand_dims(train_xdata, 0)
train_ydata = train_labels[10]

logits = cnn_dense_model(in_true, out_true, 'test')
print(sess.run(logits, feed_dict={in_true: train_xdata, out_true: train_ydata}))
'''
"""
