'''
Created on Oct 21, 2018

@author: pure
'''

import tensorflow as tf
import params
from tensorflow.contrib import layers

in_img_ph = tf.placeholder("float", [None, 224, 224, 3])
# label_img_ph = tf.placeholder(tf.float32, [])

# ************* #
#    Layers    #
# ************* #
with tf.variable_scope("VGG16"):
    for i in range(13):
        pre_idx = i
        current_idx = i+1
        
        # ******************** #
        # Convolutional Layers #
        # ******************** #
        with tf.variable_scope('conv_layer{0}'.format(current_idx)):
            if current_idx is 1:
                globals()['layer{0}'.format(current_idx)] = tf.nn.conv2d(in_img_ph, 
                                                                         filter=params.conv_weights["cw{0}".format(current_idx)], 
                                                                         strides=params.strides['1x1'], 
                                                                         padding='SAME')
            else:
                globals()['layer{0}'.format(current_idx)] = tf.nn.conv2d(globals()['layer{0}'.format(pre_idx)], 
                                                                         filter=params.conv_weights["cw{0}".format(current_idx)], 
                                                                         strides=params.strides['1x1'], 
                                                                         padding='SAME')
                
            globals()['layer{0}'.format(current_idx)] = tf.nn.bias_add(globals()['layer{0}'.format(current_idx)], 
                                                                       params.conv_biases["cb{0}".format(current_idx)])
            globals()['layer{0}'.format(current_idx)] = tf.nn.relu(globals()['layer{0}'.format(current_idx)])
            
            if current_idx in params.pool_layers:
                globals()['layer{0}'.format(current_idx)] = tf.nn.avg_pool(globals()['layer{0}'.format(current_idx)], 
                                                                           ksize=params.pool_size['2x2'], 
                                                                           strides=params.strides['2x2'], 
                                                                           padding='SAME')
                
            print(globals()['layer{0}'.format(current_idx)])
            
# ****************************** #
# Convolutional Transpose Layers #
# ****************************** #
    with tf.variable_scope('conv_transpose'):
        with tf.variable_scope('layer3'):
            layer3_trans = tf.nn.conv2d_transpose(globals()['layer3'], params.conv_trans_weights['ctw3'], output_shape=[-1, 224, 224, 2],
                                                   strides=params.strides['2x2'], padding='SAME')
            #layer3_trans = tf.layers.conv2d_transpose(globals()['layer3'], 2, 4, strides=(2, 2), padding='same', 
            #                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            # L2 Regularizer
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/l2_regularizer
            
            print(layer3_trans)