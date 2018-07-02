import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt

#========================================#
#========================================#
#           Data Preprocessing           #
#========================================#
#========================================#

#abs_file_name = "/Users/user170/Developments/Personal-Dev./Machine-Learning/Data/Face-SJC/employee-faces.pkl"
abs_file_name = "/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/Face-SJC/employee-faces.pkl"

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

dataset = unpickle(abs_file_name)

features = dataset['data']
labels = dataset['labels']
ref_labels = dataset['ref_labels']

print("features shape: ", features.shape)
print("labels shape: ", labels.shape)

########################################
# Dataset preprocessing: Normalization #
########################################

def scailing(img, new_min = 0, new_max = 1):
    new_img = img.copy()
    new_img = cv2.normalize(new_img, dst=None, alpha=new_min, beta=new_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return new_img

# # Normalization for object images
def normalization(dataset):
    new_features = []
    
    for f in tqdm(dataset):
        #f_gray = self.bgr2gray(f)
        #f_norm = scailing(f, new_min=0, new_max=1)
        #print("feature1 < min: {0} | max: {1} >".format(np.min(f_norm), np.max(f_norm)))
        new_features.append(scailing(f, new_min=0, new_max=1))
        
    return np.array(new_features)
    
#labels = labels.reshape(-1, 1)
features = normalization(features)
#labels = labels.reshape(-1, 1)

##########################################
# Dataset preprocessing: Onthot encoding #
##########################################

tf.one_hot(labels, labels.max()+1, axis=1)

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    labels = sess.run(tf.one_hot(labels, labels.max()+1, axis=1))

print("Onehot labels shape: ", labels.shape)

#################################################
# Dataset preprocessing: Train set and Test set #
#################################################

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.33, random_state=42)

print("(train shape) features: {0}, labels: {1}".format(features_train.shape, labels_train.shape))
print("(test shape) features: {0}, labels: {1}".format(features_test.shape, labels_test.shape))


#==================================#
#==================================#
#           Resnet model           #
#==================================#
#================================= #

######################
# network parameters #
######################

class netParam(object):
    def __init__(self, epoch = 5000, batch_size = 1000, learning_rate = 0.001):
        # Input nodes
        self.in_feature_shape = 224
        self.in_feature_channels = 3
        
        self.out_size = 9
        
        self._input_nd = tf.placeholder(
            tf.float32, shape=[None, self.in_feature_shape, self.in_feature_shape, self.in_feature_channels], name="input")
        self._labels_nd = tf.placeholder(tf.int32, shape=[None, self.out_size], name="labels")
        
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

####################
# Residual Network #
####################

class Resnet(netParam):
    def __init__(self, epoch = 5000, batch_size = 1000, learning_rate = 0.001):
        super(Resnet, self).__init__(epoch, batch_size, learning_rate)

    def identity_function(self, input_node, filter_depth, identity_name, half=False):
        with tf.variable_scope(identity_name):
            if half:
                input_node = self._convolutional_layer(input_node, self.kernel_3x3, filter_depth, self.stride_2x2, "SAME", "conv_in", False)
                #print(input_node)
            conv1 = self._convolutional_layer(input_node, self.kernel_3x3, filter_depth, self.stride_1x1, "SAME", "conv1", False)
            #print(conv1)
            relu = tf.nn.relu(conv1)
            #print(relu)
            conv2 = self._convolutional_layer(relu, self.kernel_3x3, filter_depth, self.stride_1x1, "SAME", "conv2", False)
            #print(conv2)
            
            #conv2 = tf.multiply(conv2, 0.1)
            conv2 = tf.multiply(conv2, 0.01)
            skip = tf.add(input_node, conv2, name="skip")
            #skip = tf.multiply(input_node, conv2, name="skip")
            relu = tf.nn.relu(skip)
            
            return relu
    def simple_CNN(self, input_node):
        # CONVOLUTIONAL 1
        conv1 = self._convolutional_layer(input_node, 3, 32, 1, "VALID", "conv1")
        
        # FLATTEN
        relu1_shape = conv1.get_shape().as_list()
        conv1_flat = tf.reshape(conv1, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])
        
        # DENSE 1
        dense1 = self._dense_layer(conv1_flat, 128, "relu", "dense1")

        # DENSE 2
        dense2 = self._dense_layer(dense1, 10, "None", "dense2")
        
        return dense2
        
    def resnet(self, input_node):
        print("|============================== MODEL INFO. ====================================|")
        print("  Input  : ", input_node)
        # Convolutional layer 1: conv-> relu -> max_pool
        #with tf.variable_scope("layer_1"):
        layer_1 = self._convolutional_layer(input_node, self.kernel_3x3, self.depth_64, self.stride_1x1, "SAME", "layer_1", False)
        #print("  Layer 1: ", layer_1)
        layer_1 = tf.nn.relu(layer_1, name="layer_1/Relu")
        #print("  Layer 1: ", layer_1)
        layer_1 = self._max_pool(layer_1, self.kernel_2x2, self.stride_2x2, "VALID", "layer_1")
        print("  Layer 1: ", layer_1)
        
        layer_2 = self.identity_function(layer_1, 64, "layer_2")
        print("  Layer 2: ", layer_2)
        
        layer_3 = self.identity_function(layer_2, 64, "layer_3")
        print("  Layer 3: ", layer_3)
        
        layer_4 = self.identity_function(layer_3, 64, "layer_4")
        print("  Layer 4: ", layer_4)
        
        layer_5 = self.identity_function(layer_4, self.depth_128, "layer_5", True)
        print("  Layer 5: ", layer_5)
        
        layer_6 = self.identity_function(layer_5, self.depth_128, "layer_6")
        print("  Layer 6: ", layer_6)
        
        layer_7 = self.identity_function(layer_5, self.depth_128, "layer_7")
        print("  Layer 7: ", layer_7)
        
        layer_8 = self.identity_function(layer_5, self.depth_128, "layer_8")
        print("  Layer 8: ", layer_8)
        
        layer_8_avg_pool = self._avg_pool(layer_8, self.kernel_2x2, self.stride_2x2, "VALID", "layer_8")
        print("  Layer 8: ", layer_8_avg_pool)
        
        layer_8_pool_shape = layer_8_avg_pool.get_shape().as_list()
        layer_8_flat = tf.reshape(layer_8_avg_pool, [-1, layer_8_pool_shape[1]*layer_8_pool_shape[2]*layer_8_pool_shape[3]])
        print("  Layer 8: ", layer_8_flat)
        
        #layer_8_flat_shape = layer_8_flat.get_shape().as_list()
        logits = self._dense_layer(layer_8_flat, self.out_size, "None", name="logits")
        print("  Logits : ", logits)
        print("|===============================================================================|")
        
        return logits

    def _cost_function(self, logits):
        # CONST FUNCTION
        #loss_nd = tf.losses.sparse_softmax_cross_entropy(labels=self._labels_nd, logits=logits)
        print("KKK: ", logits)
        loss_nd = tf.nn.softmax_cross_entropy_with_logits(labels=self._labels_nd, logits=logits)
        return loss_nd

    def _optimizer(self, loss):
        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate=self._rl)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op

    def _prediction(self, logits):
        # PREDICTION
        pred = tf.argmax(logits, 1)
        return pred

    def _accuracy(self, pred):
        # ACCURACY
        acc = tf.equal(pred, tf.argmax(self._labels_nd, 1))
        return acc
        
    def _accuracy_info(self, iterator, session, accuracy_node, batch_x, batch_y, test_x, test_y):
        train_acc = session.run(accuracy_node, feed_dict={self._input_nd: batch_x, self._labels_nd: batch_y})
        
        batch_index = np.random.choice(len(test_x), size=self._batch_size)
        test_batch_x = test_x[batch_index]
        test_batch_y = test_y[batch_index]
        
        eval_acc = session.run(accuracy_node, feed_dict={self._input_nd: test_batch_x, self._labels_nd: test_batch_y})

        #print("({0}/{1}) TRAIN ACC.: {2} %"\
        #      .format(iter, self._epoch, np.sum(train_acc)/len(train_acc)*100))

        print("({0}/{1}) TRAIN ACC.: {2} % | EVAL ACC.: {3} %"\
              .format(iterator, self._epoch, np.sum(train_acc)/len(train_acc)*100, np.sum(eval_acc)/len(eval_acc)*100))
        
    def train(self):
        logits = self.resnet(self._input_nd)
        #logits = self.simple_CNN(self._input_nd)
        loss = self._cost_function(logits)
        optimizer = self._optimizer(loss)
        pred = self._prediction(logits)
        acc = self._accuracy(pred)
        
        fig, ax = plt.subplots(5, 5, figsize=(20, 20))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for iterator in range(self._epoch):
                batch_index = np.random.choice(len(features_train), size=self._batch_size)
                #self.batch_x = features[batch_index]
                #self.batch_y = labels[batch_index]

                batch_x = features_train[batch_index]
                batch_y = labels_train[batch_index]

                '''
                a = sess.run(logits, feed_dict={self._input_nd: batch_x, self._labels_nd: batch_y})
                print(np.shape(a))
                '''
                sess.run(optimizer, feed_dict={self._input_nd: batch_x, self._labels_nd: batch_y})

                if iterator % 10 == 0:
                    self._accuracy_info(iterator, sess, acc, batch_x, batch_y, features_test, labels_test)
                    
                    pred_labels = sess.run(pred, feed_dict={self._input_nd: features_test[:self._batch_size]})
                    print("\t", np.argmax(labels_test[:self._batch_size][:10], axis=1))
                    print("\t", pred_labels[:10], "\n")
                    #print(np.sum(pred_labels != 0))
                
                for i in range(5):
                    for j in range(5):
                        ax[i, j].imshow(features_test[i*11+j])
                        ax[i, j].set_title(ref_labels[pred_labels[i*11+j]])
                
                plt.tight_layout()
                plt.pause(0.0001)
                plt.draw()
                
                '''
                for i in range(4):
                    ax[i].imshow(features_test[:self._batch_size][i])
                    ax[i].set_title(pred_labels[i])

                plt.pause(0.001)
                plt.show()
                '''

if __name__ == "__main__":
    rn = Resnet(epoch = 1000, batch_size = 100, learning_rate = 0.001)
    rn.train()