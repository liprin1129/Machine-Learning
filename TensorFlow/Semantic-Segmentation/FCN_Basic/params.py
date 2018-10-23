'''
Created on Oct 16, 2018

@author: pure
'''

import tensorflow as tf
from tensorflow.contrib import layers

with tf.variable_scope("VGG16"):
    conv_weights = {
        'cw1': tf.get_variable('CW1', shape=(3,3,3,64), initializer=layers.xavier_initializer()),
        'cw2': tf.get_variable('CW2', shape=(3,3,64,64), initializer=layers.xavier_initializer()),
        'cw3': tf.get_variable('CW3', shape=(3,3,64,128), initializer=layers.xavier_initializer()),
        'cw4': tf.get_variable('CW4', shape=(3,3,128,128), initializer=layers.xavier_initializer()),
        'cw5': tf.get_variable('CW5', shape=(3,3,128,256), initializer=layers.xavier_initializer()),
        'cw6': tf.get_variable('CW6', shape=(3,3,256,256), initializer=layers.xavier_initializer()),
        'cw7': tf.get_variable('CW7', shape=(1,1,256,256), initializer=layers.xavier_initializer()),
        'cw8': tf.get_variable('CW8', shape=(3,3,256,512), initializer=layers.xavier_initializer()),
        'cw9': tf.get_variable('CW9', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw10': tf.get_variable('CW10', shape=(1,1,512,512), initializer=layers.xavier_initializer()),
        'cw11': tf.get_variable('CW11', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw12': tf.get_variable('CW12', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw13': tf.get_variable('CW13', shape=(1,1,512,512), initializer=layers.xavier_initializer())
        }
    
    conv_biases = {
        'cb1': tf.get_variable('CB1', shape=(64), initializer=layers.xavier_initializer()),
        'cb2': tf.get_variable('CB2', shape=(64), initializer=layers.xavier_initializer()),
        'cb3': tf.get_variable('CB3', shape=(128), initializer=layers.xavier_initializer()),
        'cb4': tf.get_variable('CB4', shape=(128), initializer=layers.xavier_initializer()),
        'cb5': tf.get_variable('CB5', shape=(256), initializer=layers.xavier_initializer()),
        'cb6': tf.get_variable('CB6', shape=(256), initializer=layers.xavier_initializer()),
        'cb7': tf.get_variable('CB7', shape=(256), initializer=layers.xavier_initializer()),
        'cb8': tf.get_variable('CB8', shape=(512), initializer=layers.xavier_initializer()),
        'cb9': tf.get_variable('CB9', shape=(512), initializer=layers.xavier_initializer()),
        'cb10': tf.get_variable('CB10', shape=(512), initializer=layers.xavier_initializer()),
        'cb11': tf.get_variable('CB11', shape=(512), initializer=layers.xavier_initializer()),
        'cb12': tf.get_variable('CB12', shape=(512), initializer=layers.xavier_initializer()),
        'cb13': tf.get_variable('CB13', shape=(512), initializer=layers.xavier_initializer())
        }
    
    conv_trans_weights = {
        'ctw3': tf.get_variable('CTW3', shape=(4,4,2,128), initializer=layers.xavier_initializer())
        }
    
    strides = {
        '1x1': [1, 1, 1, 1],
        '2x2': [1, 2, 2, 1],
        '8x8': [1, 8, 8, 1]
    }
    
    pool_layers = [2, 4, 7, 10, 13]
    
    pool_size = {
        '1x1': [1, 1, 1, 1],
        '2x2': [1, 2, 2, 1]
    }