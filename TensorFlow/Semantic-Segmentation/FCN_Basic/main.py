'''
Created on Jul 26, 2018

@author: pure
'''

import tensorflow as tf
from function_wrapper import convolution_2d

in_img_ph = tf.placeholder("float", [None, 224, 224, 3])
# label_img_ph = tf.placeholder(tf.float32, [])

# ************* #
#    Layer 1    #
# ************* #

# ** Variable and Constants setup ** #
weight1_vb = tf.Variable(
    tf.truncated_normal([3, 3, 3, 64], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw1"))

bias1_vb = tf.Variable(
    tf.truncated_normal([64], name='cb1'))

strides_conv1 = [1, 1, 1, 1]

# ** Operations ** #
conv1 = convolution_2d(in_img_ph, weight1_vb, bias1_vb, strides_conv1, 'SAME', True, 'layer1')

print(conv1)

# ************* #
#    Layer 2    #
# ************* #

# ** Variable and Constants setup ** #
weight2_vb = tf.Variable(
    tf.truncated_normal([3, 3, 64, 64], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw2"))

bias2_vb = tf.Variable(
    tf.truncated_normal([64], name='cb2'))

strides_conv2 = [1, 1, 1, 1]

# ** Operations ** #
conv2 = convolution_2d(conv1, weight2_vb, bias2_vb, strides_conv2, 'SAME', True, 'layer2')
conv2_pooling = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print(conv2_pooling)

# ************* #
#    Layer 3    #
# ************* #

# ** Variable and Constants setup ** #
weight3_vb = tf.Variable(
    tf.truncated_normal([3, 3, 64, 128], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw3"))

bias3_vb = tf.Variable(
    tf.truncated_normal([128], name='cb3'))

strides_conv3 = [1, 1, 1, 1]

# ** Operations ** #
conv3 = convolution_2d(conv2_pooling, weight3_vb, bias3_vb, strides_conv3, 'SAME', True, 'layer3')

print(conv3)

# ************* #
#    Layer 4    #
# ************* #

# ** Variable and Constants setup ** #
weight4_vb = tf.Variable(
    tf.truncated_normal([3, 3, 128, 128], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw4"))

bias4_vb = tf.Variable(
    tf.truncated_normal([128], name='cb4'))

strides_conv4 = [1, 1, 1, 1]

# ** Operations ** #
conv4 = convolution_2d(conv3, weight4_vb, bias4_vb, strides_conv4, 'SAME', True, 'layer4')
conv4_pooling = tf.nn.avg_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print(conv4_pooling)

# ************* #
#    Layer 5    #
# ************* #

# ** Variable and Constants setup ** #
weight5_vb = tf.Variable(
    tf.truncated_normal([3, 3, 128, 256], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw5"))

bias5_vb = tf.Variable(
    tf.truncated_normal([256], name='cb5'))

strides_conv5 = [1, 1, 1, 1]

# ** Operations ** #
conv5 = convolution_2d(conv4_pooling, weight5_vb, bias5_vb, strides_conv5, 'SAME', True, 'layer5')

print(conv5)

# ************* #
#    Layer 6    #
# ************* #

# ** Variable and Constants setup ** #
weight6_vb = tf.Variable(
    tf.truncated_normal([3, 3, 256, 256], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw6"))

bias6_vb = tf.Variable(
    tf.truncated_normal([256], name='cb6'))

strides_conv6 = [1, 1, 1, 1]

# ** Operations ** #
conv6 = convolution_2d(conv5, weight6_vb, bias6_vb, strides_conv6, 'SAME', True, 'layer6')

print(conv6)

# ************* #
#    Layer 7    #
# ************* #

# ** Variable and Constants setup ** #
weight7_vb = tf.Variable(
    tf.truncated_normal([1, 1, 256, 256], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw7"))

bias7_vb = tf.Variable(
    tf.truncated_normal([256], name='cb7'))

strides_conv7 = [1, 1, 1, 1]

# ** Operations ** #
conv7 = convolution_2d(conv6, weight7_vb, bias7_vb, strides_conv7, 'SAME', True, 'layer7')
conv7_pooling = tf.nn.avg_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv7)

# ************* #
#    Layer 8    #
# ************* #

# ** Variable and Constants setup ** #
weight8_vb = tf.Variable(
    tf.truncated_normal([3, 3, 256, 512], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw8"))

bias8_vb = tf.Variable(
    tf.truncated_normal([512], name='cb8'))

strides_conv8 = [1, 1, 1, 1]

# ** Operations ** #
conv8 = convolution_2d(conv7_pooling, weight8_vb, bias8_vb, strides_conv8, 'SAME', True, 'layer8')

# ************* #
#    Layer 9    #
# ************* #

# ** Variable and Constants setup ** #
weight9_vb = tf.Variable(
    tf.truncated_normal([3, 3, 512, 512], mean=0, stddev=0.03, dtype=tf.float32, seed=10, name="cw9"))

bias9_vb = tf.Variable(
    tf.truncated_normal([512], name='cb9'))

strides_conv9 = [1, 1, 1, 1]

# ** Operations ** #
conv9 = convolution_2d(conv8, weight9_vb, bias9_vb, strides_conv9, 'SAME', True, 'layer9')
#if __name__ == '__main__':