'''
Created on Jun 19, 2018

@author: user170
'''

import tensorflow as tf

class netParam(object):
    '''
    classdocs
    '''
    def __init__(self, epoch = 5000, batch_size = 1000, learning_rate = 0.001):
        # Input nodes
        self.in_feature_shape = 32
        self.in_feature_channels = 3
        
        self._input_nd = tf.placeholder(
            tf.float32, shape=[None, self.in_feature_shape, self.in_feature_shape, self.in_feature_channels], name="input")
        self._labels_nd = tf.placeholder(tf.int32, shape=[None, 10], name="labels")
        
        '''
        # Layer nodes
        self._logits_nd = None
        self._loss_nd = None
        self._train_optimizer_nd = None
        self._pred_nd = None
        self._acc_nd = None
        '''
        
        # Hyper parameters
        self.kernel_1x1 = 1
        self.kernel_2x2 = 2
        self.kernel_3x3 = 3
        
        self.depth_64 = 64
        self.depth_128 = 128
        self.layer_3_depth = 48
        self.layer_4_depth = 96
        self.layer_5_depth = 192

        self.dense_4096 = 4096
        self.dense_1000 = 1000
        self.dense_output = 2

        self.stride_1x1 = 1
        self.stride_2x2 = 2
        
        # Training parameters
        self._epoch = epoch
        self._batch_size = batch_size
        self._rl = learning_rate
    
    @classmethod
    def _filter_var(cls, kernel, in_depth, out_depth, node_name):
        return tf.Variable(
            tf.truncated_normal((kernel, kernel, in_depth, out_depth), stddev=0.1, dtype=tf.float32),
            name=node_name+"_filter")
    
    @classmethod
    def _bias_var(cls, size, node_name):
        return tf.Variable(
            tf.zeros([size], dtype=tf.float32), 
            name=node_name+"_bias")

    @classmethod
    def _dense_var(cls, in_size, out_size, node_name):
        return tf.Variable(
            tf.truncated_normal([in_size, out_size], stddev=0.1, dtype=tf.float32),
            name=node_name+"_dense")
    
    @classmethod
    def _max_pool(cls, in_node, kernel_size, stride_size, pad_type, node_name):
        return tf.nn.max_pool(in_node, 
                              ksize=[1, kernel_size, kernel_size, 1], 
                              strides=[1, stride_size, stride_size, 1], 
                              padding=pad_type,
                              name=node_name+"/Max_pool")
    
    @classmethod
    def _avg_pool(cls, in_node, kernel_size, stride_size, pad_type, node_name):
        return tf.nn.avg_pool(in_node, 
                              ksize=[1, kernel_size, kernel_size, 1], 
                              strides=[1, stride_size, stride_size, 1], 
                              padding=pad_type,
                              name=node_name+"/Avg_pool")
    
    def _convolutional_layer(self, in_node, kernel_size, out_depth, stride, padding, name, use_bias=False):
        with tf.variable_scope(name):
            # Weight to be learned
            conv_filter = self._filter_var(
                kernel = kernel_size,
                in_depth=in_node.get_shape().as_list()[-1],
                out_depth=out_depth,
                node_name=name)
            
            # Convolutional node
            conv = tf.nn.conv2d(in_node, conv_filter, [1, stride, stride, 1], padding=padding)
            
            # Biase to be learned
            if use_bias:
                conv_bias = self._bias_var(size=out_depth, node_name=name)
                conv = tf.nn.bias_add(conv, conv_bias)
            
            #conv = tf.nn.relu(conv)

            return conv

    def _dense_layer(self, in_node, dense_size, relu, name):
        #print(in_node.get_shape().as_list())
        dense_weight = self._dense_var(in_size=in_node.get_shape().as_list()[-1], out_size=dense_size, node_name=name)
        dense_bias = self._bias_var(size=dense_size, node_name=name)

        if relu == "relu":
            return tf.nn.relu(tf.add(tf.matmul(in_node, dense_weight), dense_bias))
        else:
            return tf.add(tf.matmul(in_node, dense_weight), dense_bias)