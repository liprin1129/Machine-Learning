# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:38:40 2018

@author: user170
"""

#import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))
my_output = tf.multiply(x_data, A)

loss = tf.square(my_output - y_target)

my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_stochastic = []
for i in xrange(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    
    if (i+1)%5==0:
        print 'Step #' + str(i+1) + ' A = ' + str(sess.run(A))
        temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
        print 'Loss = ' + str(temp_loss)
        loss_stochastic.append(temp_loss)

'''
merged = tf.summary.merge_all(key='summaries')
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')
    
my_writer = tf.summary.FileWriter('tensorboard_logs/', tf.get_default_graph())
'''