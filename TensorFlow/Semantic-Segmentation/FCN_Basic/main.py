'''
Created on Jul 26, 2018

@author: pure
'''

import tensorflow as tf
import params

in_img_ph = tf.placeholder("float", [None, 224, 224, 3])
# label_img_ph = tf.placeholder(tf.float32, [])

# ************* #
#    Layer 1    #
# ************* #

with tf.variable_scope("VGG16"):
    with tf.variable_scope("layer1"):
        # ** Operations ** #
        layer1 = tf.nn.conv2d(in_img_ph, filter=params.weights["CW1"], strides=params.strides['1x1'], padding='SAME')
        layer1= tf.nn.bias_add(layer1, params.biases["CB1"])
        layer1 = tf.nn.relu(layer1)
        
        print(layer1)

# ************* #
#    Layer 2    #
# ************* #

    with tf.variable_scope("layer2"):
    # ** Operations ** #
        layer2 = tf.nn.conv2d(layer1, filter=params.weights["CW2"], strides=params.strides['1x1'])
        layer2 = tf.nn.bias_add(layer2, params.biases["CB2"])
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.nn.avg_pool(layer2, ksize=params.pool_size['2x2'], strides=params.strides['2x2'])
        
        print(layer2)

#if __name__ == '__main__':