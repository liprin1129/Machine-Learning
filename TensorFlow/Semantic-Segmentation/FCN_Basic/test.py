'''
Created on Oct 29, 2018

@author: pure
'''

import tensorflow as tf
import params

with tf.variable_scope("Group1"):
    layer1 = tf.layers.conv2d(inputs=params.input_ph, filters=params.kernel_depth['64'], 
                              kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                              padding='same',activation=tf.nn.relu, use_bias=True, name='layer1')
    print(layer1)
    
    layer2 = tf.layers.conv2d(inputs=layer1, filters=params.kernel_depth['64'], 
                              kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                              padding='same',activation=tf.nn.relu, use_bias=True, name='layer2')
    print(layer2)
    
    layer3 = tf.layers.conv2d(inputs=layer2, filters=params.kernel_depth['64'], 
                              kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                              padding='same',activation=tf.nn.relu, use_bias=True, name='layer3')
    print(layer3)
    
    layer4 = tf.nn.avg_pool(layer3, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer4')
    print(layer4)
    
    layer5 = tf.layers.conv2d_transpose(inputs=layer4, filters=2, kernel_size=(3, 3), strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='skip_layer4')
    
    #layer5 = tf.nn.conv2d_transpose(layer4, filter=params.conv_trans_weights['ctw2_add'], 
    #                                             output_shape=tf.shape(params.input_ph), strides=params.conv_strides['2x2'], padding='SAME')
    print(layer5)