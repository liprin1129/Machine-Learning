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
        layer1 = tf.nn.conv2d(in_img_ph, filter=params.weights["cw1"], strides=params.strides['1x1'], padding='SAME')
        layer1= tf.nn.bias_add(layer1, params.biases["cb1"])
        layer1 = tf.nn.relu(layer1)
        
        print(layer1)

# ************* #
#    Layer 2    #
# ************* #
    with tf.variable_scope("layer2"):
    # ** Operations ** #
        layer2 = tf.nn.conv2d(layer1, filter=params.weights["cw2"], strides=params.strides['1x1'], padding='SAME')
        layer2 = tf.nn.bias_add(layer2, params.biases["cb2"])
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.nn.avg_pool(layer2, ksize=params.pool_size['2x2'], strides=params.strides['2x2'], padding='SAME')
        
        print(layer2)
        
# ************* #
#    Layer 3    #
# ************* #
    with tf.variable_scope("layer3"):
    # ** Operations ** #
        layer3 = tf.nn.conv2d(layer2, filter=params.weights["cw3"], strides=params.strides['1x1'], padding='SAME')
        layer3 = tf.nn.bias_add(layer3, params.biases["cb3"])
        layer3 = tf.nn.relu(layer3)
        
        print(layer3)
        
# ************* #
#    Layer 4    #
# ************* #
    with tf.variable_scope("layer4"):
    # ** Operations ** #
        layer4 = tf.nn.conv2d(layer3, filter=params.weights["cw4"], strides=params.strides['1x1'], padding='SAME')
        layer4 = tf.nn.bias_add(layer4, params.biases["cb4"])
        layer4 = tf.nn.relu(layer4)
        layer4 = tf.nn.avg_pool(layer4, ksize=params.pool_size['2x2'], strides=params.strides['2x2'], padding='SAME')
        
        print(layer4)
        
# ************* #
#    Layer 5    #
# ************* #
    with tf.variable_scope("layer5"):
    # ** Operations ** #
        layer5 = tf.nn.conv2d(layer4, filter=params.weights["cw5"], strides=params.strides['1x1'], padding='SAME')
        layer5 = tf.nn.bias_add(layer5, params.biases["cb5"])
        layer5 = tf.nn.relu(layer5)
        
        print(layer5)

# ************* #
#    Layer 6    #
# ************* #
    with tf.variable_scope("layer6"):
    # ** Operations ** #
        layer6 = tf.nn.conv2d(layer5, filter=params.weights["cw6"], strides=params.strides['1x1'], padding='SAME')
        layer6 = tf.nn.bias_add(layer6, params.biases["cb6"])
        layer6 = tf.nn.relu(layer6)
        
        print(layer6)
        
# ************* #
#    Layer 7    #
# ************* #
    with tf.variable_scope("layer7"):
    # ** Operations ** #
        layer7 = tf.nn.conv2d(layer6, filter=params.weights["cw7"], strides=params.strides['1x1'], padding='SAME')
        layer7 = tf.nn.bias_add(layer7, params.biases["cb7"])
        layer7 = tf.nn.relu(layer7)
        layer7 = tf.nn.avg_pool(layer7, ksize=params.pool_size['2x2'], strides=params.strides['2x2'], padding='SAME')
        
        print(layer7)
        
# ************* #
#    Layer 8    #
# ************* #
    with tf.variable_scope("layer8"):
    # ** Operations ** #
        layer8 = tf.nn.conv2d(layer7, filter=params.weights["cw8"], strides=params.strides['1x1'], padding='SAME')
        layer8 = tf.nn.bias_add(layer8, params.biases["cb8"])
        layer8 = tf.nn.relu(layer8)
        
        print(layer8)
        
# ************* #
#    Layer 9    #
# ************* #
    with tf.variable_scope("layer9"):
    # ** Operations ** #
        layer9 = tf.nn.conv2d(layer8, filter=params.weights["cw9"], strides=params.strides['1x1'], padding='SAME')
        layer9 = tf.nn.bias_add(layer9, params.biases["cb9"])
        layer9 = tf.nn.relu(layer9)
        
        print(layer6)
        
# ************* #
#    Layer 10    #
# ************* #
    with tf.variable_scope("layer10"):
    # ** Operations ** #
        layer10 = tf.nn.conv2d(layer9, filter=params.weights["cw10"], strides=params.strides['1x1'], padding='SAME')
        layer10 = tf.nn.bias_add(layer10, params.biases["cb10"])
        layer10 = tf.nn.relu(layer10)
        layer10 = tf.nn.avg_pool(layer10, ksize=params.pool_size['2x2'], strides=params.strides['2x2'], padding='SAME')
        
        print(layer7)
        
# ************* #
#    Layer 11    #
# ************* #
    with tf.variable_scope("layer11"):
    # ** Operations ** #
        layer11 = tf.nn.conv2d(layer10, filter=params.weights["cw11"], strides=params.strides['1x1'], padding='SAME')
        layer11 = tf.nn.bias_add(layer11, params.biases["cb11"])
        layer11 = tf.nn.relu(layer11)
        
        print(layer11)
        

#if __name__ == '__main__':