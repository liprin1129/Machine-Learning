# IMPORT TENSORFLOW
import tensorflow as tf
#from tensorflow.contrib.layers import flatten

# IMPORT NECCESSARY PACKAGES
import numpy as np
from tqdm import tqdm

from data_wrangling import PickleHelper

class net_setup(object):
    def __init__(self):
        # Net info.
        self.learning_rate = 0.5
        self.epoch = 5000
        self.batch_size = 50
        self.keep_prob_node = tf.placeholder(tf.float32)
        
        # Network shapes
        self.kernel_1x1 = 1
        self.kernel_3x3 = 3
        
        self.depth1 = 12
        self.depth2 = 24
        self.depth3 = 48
        self.depth4 = 96
        self.depth5 = 192

        self.dense_4096 = 4096
        self.dense_1000 = 1000
        self.dense_output = 2

    @classmethod
    def filter_var(cls, kernel, in_depth, out_depth, node_name):
        return tf.Variable(
            tf.truncated_normal((kernel, kernel, in_depth, out_depth), stddev=0.1, dtype=tf.float32),
            name=node_name+"_filter")
    
    def bias_var(cls, size, node_name):
        return tf.Variable(
            tf.zeros([size], dtype=tf.float32), 
            name=node_name+"_bias")

    def dense_var(cls, in_size, out_size, node_name):
        return tf.Variable(
            tf.truncated_normal([in_size, out_size], stddev=0.1, dtype=tf.float32),
            name=node_name+"_dense")
    
    def max_pool(cls, in_node, node_name):
        return tf.nn.max_pool(in_node, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=node_name)

class VGG16(net_setup):
    def __init__(self, path, feature_name, label_name):
        super(VGG16, self).__init__()

        #self.__features, self.__feature_shape = self.reshape_for_tensor(features)
        #self.__labels = labels
        #self.__labels_shape = (None, np.shape(labels)[1])
        self.__features = PickleHelper.load_pickle(path, feature_name)
        self.__labels = PickleHelper.load_pickle(path, label_name)

        # For networks
        self.__input_nd = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.__label_nd = tf.placeholder(tf.float32, shape=[None, 2])
        
        self.__logits_nd = None
        self.__cost_function_nd = None
        self.__optimizer_nd = None
        self.__correct_prediction_nd = None
        self.__accuracy_nd = None
        self.__pred_nd = None # Prediction node
        self.__acc_nd = None # Accuracy node

    
    '''
    def input_node(self, data_shape):       
        return tf.placeholder(tf.float32, shape=data_shape)
    '''

    def convolutional_node(self, in_node, kernel_size, out_depth, stride, padding, name):
        with tf.variable_scope(name):
            conv_filter = self.filter_var(
                kernel = kernel_size,
                in_depth=in_node.get_shape().as_list()[-1],
                out_depth=out_depth,
                node_name=name)
            #print(conv_filter.shape)
            conv_bias = self.bias_var(size=out_depth, node_name=name)
            #print(conv_bias.shape)

            conv = tf.nn.conv2d(in_node, conv_filter, [1, stride, stride, 1], padding=padding)
            conv = tf.nn.bias_add(conv, conv_bias)
            conv = tf.nn.relu(conv)

            return conv

    def dense_node(self, in_node, dense_size, name):
        #print(in_node.get_shape().as_list())
        dense_weight = self.dense_var(in_size=in_node.get_shape().as_list()[-1], out_size=dense_size, node_name=name)
        dense_bias = self.bias_var(size=dense_size, node_name=name)
        return tf.nn.relu(tf.add(tf.matmul(in_node, dense_weight), dense_bias))

    def VGG16(self):
        conv1 = self.convolutional_node(self.__input_nd, 3, 32, 1, "VALID", "conv1")

        # FLATTEN
        conv1_shape = conv1.get_shape().as_list()
        conv1_flat = tf.reshape(conv1, [-1, conv1_shape[1]*conv1_shape[2]*conv1_shape[3]])

        dense1 = self.dense_node(conv1_flat, 128, "dense1")
        dense2 = self.dense_node(dense1, 2, "dense2")

        self.__logits_nd = dense2

    def __loss(self, loss_type, logits_node, hot_encoded_labels_node):
        if loss_type == "softmax":
            self.__cost_function_nd = tf.losses.softmax_cross_entropy(onehot_labels=hot_encoded_labels_node, logits=logits_node)
                                            
        elif loss_type == "sigmoid":
            self.__cost_function_nd = tf.reduce_mean(
                #tf.nn.softmax_cross_entropy_with_logits(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_from_network_node, labels=hot_encoded_labels_node))

    def __optimizer(self, loss_node, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.__optimizer_nd = optimizer.minimize(loss=loss_node, global_step=tf.train.get_global_step())

    def __prediction(self, logits_node):
        self.__pred_nd = tf.argmax(logits_node, 1)

    def __accuracy(self, pred_node, hot_encoded_labels_node):
        self.__acc_nd = tf.equal(pred_node, tf.argmax(hot_encoded_labels_node, 1))

    def run(self):
        self.VGG16()
        print(self.__logits_nd)

        self.__loss("softmax", self.__logits_nd, self.__label_nd)
        self.__optimizer(self.__cost_function_nd, self.learning_rate)
        self.__prediction(self.__logits_nd)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for iter in tqdm(range(self.epoch)):
                batch_index = np.random.choice(len(self.__features), size=self.batch_size)
                batch_x = self.__features[batch_index]
                batch_y = self.__labels[batch_index]

                sess.run(self.__optimizer_nd, feed_dict={self.__input_nd: batch_x, self.__label_nd: batch_y})

                if iter % 100 == 0:            
                    pred = sess.run(self.__pred_nd, feed_dict={self.__input_nd: batch_x})
                    print(pred)


if __name__ == "__main__":
    vgg16 = VGG16("../../Data/Face/" , "faces-obj-32x32-features-norm.pkl", "faces-obj-32x32-labels-norm.pkl")
    vgg16.run()
