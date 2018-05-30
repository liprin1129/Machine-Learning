from data_wrangling import PickleHelper
import tensorflow as tf

class netParam(object):
    def __init__(self, epoch = 500, batch_size = 500, learning_rate = 0.001):
        self._epoch = epoch
        self._batch_size = batch_size
        self._rl = learning_rate
        
        self._data = PickleHelper.load_pickle("../../../Data/Face/", "faces-obj-32x32-features-norm.pkl")
        self._labels = PickleHelper.load_pickle("../../../Data/Face/", "faces-obj-32x32-labels-norm.pkl")
        self._test_data = PickleHelper.load_pickle("../../../Data/Face/", "blob-itamochi-32x32.pkl")
        
        print("DATA: ", self._data.shape)
        
        # FOR CODE TESTING
        self._train_data = self._data[:100]
        self._train_labels = self._labels[:100]
        self._eval_data = self._data[100:120]
        self._eval_labels = self._labels[100:120]
        
        # CODE IMPLEMENTATION
        self._train_data = self._data[:80000]
        self._train_labels = self._labels[:80000]
        #eval_data = data[80000:]
        #eval_labels = labels[80000:]

    def _filter_var(cls, kernel, in_depth, out_depth, node_name):
        return tf.Variable(
            tf.truncated_normal((kernel, kernel, in_depth, out_depth), stddev=0.1, dtype=tf.float32),
            name=node_name+"_filter")
    
    def _bias_var(cls, size, node_name):
        return tf.Variable(
            tf.zeros([size], dtype=tf.float32), 
            name=node_name+"_bias")

    def _dense_var(cls, in_size, out_size, node_name):
        return tf.Variable(
            tf.truncated_normal([in_size, out_size], stddev=0.1, dtype=tf.float32),
            name=node_name+"_dense")
    
    def _max_pool(cls, in_node, node_name):
        return tf.nn.max_pool(in_node, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=node_name+"_max_pool")
