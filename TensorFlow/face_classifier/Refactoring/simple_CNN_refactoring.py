import tensorflow as tf

from tqdm import tqdm
import numpy as np

import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

from net_param import netParam
    
class CNN(netParam):
    def __init__(self):
        super(CNN, self).__init__()

        self._feature_nd = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self._label_nd = tf.placeholder(tf.float32, shape=[None, 2])
        
        self.img_channel = 3
        
        self._logits_nd = None
        self._loss_nd = None
        self._train_optimizer_nd = None
        self._pred_nd = None
        self._acc_nd = None

    def _convolutional_layer(self, in_node, kernel_size, out_depth, stride, padding, name):
        with tf.variable_scope(name):
            conv_filter = self._filter_var(
                kernel = kernel_size,
                in_depth=in_node.get_shape().as_list()[-1],
                out_depth=out_depth,
                node_name=name)
            #print(conv_filter.shape)
            conv_bias = self._bias_var(size=out_depth, node_name=name)
            #print(conv_bias.shape)

            conv = tf.nn.conv2d(in_node, conv_filter, [1, stride, stride, 1], padding=padding)
            conv = tf.nn.bias_add(conv, conv_bias)
            conv = tf.nn.relu(conv)

            return conv

    def _dense_layer(self, in_node, dense_size, relu, name):
        #print(in_node.get_shape().as_list())
        dense_weight = self._dense_var(in_size=in_node.get_shape().as_list()[-1], out_size=dense_size, node_name=name)
        dense_bias = self._bias_var(size=dense_size, node_name=name)

        if relu == "relu":
            return tf.nn.relu(tf.add(tf.matmul(in_node, dense_weight), dense_bias))
        else:
            return tf.add(tf.matmul(in_node, dense_weight), dense_bias)

    def architecture(self, input_node):
        # CONVOLUTIONAL 1
        '''
        conv1_W = tf.Variable(
            tf.truncated_normal([3, 3, 3, 32], stddev=0.1, dtype=tf.float32))
        conv1_b = tf.Variable(tf.truncated_normal([32], stddev=0.1, dtype=tf.float32))
        
        conv1 = tf.layers.conv2d(inputs=input_node, filters=32, kernel_size=[3, 3], strides=(1, 1),  padding="valid", activation=tf.nn.relu)
        '''
        conv1 = self._convolutional_layer(input_node, 3, 32, 1, "VALID", "conv1")
        #conv1 = tf.nn.conv2d(feature_node, conv1_W, strides=[1, 1, 1, 1], padding='VALID')
        #relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
        #pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)
        
        # FLATTEN
        relu1_shape = conv1.get_shape().as_list()
        conv1_flat = tf.reshape(conv1, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])
        
        # DENSE 1
        #dense1 = tf.layers.dense(inputs=conv1_flat, units=128, activation=tf.nn.relu)#tf.add(tf.matmul(
        dense1 = self._dense_layer(conv1_flat, 128, "relu", "dense1")
        # DENSE 2
        #dense2 = tf.layers.dense(inputs=dense1, units=2)#tf.add(tf.matmul(
        dense2 = self._dense_layer(dense1, 2, "None", "dense2")
        
        self._logits_nd = dense2

    def cost_function(self):
        # CONST FUNCTION
        self._loss_nd = tf.losses.softmax_cross_entropy(onehot_labels=self._label_nd, logits=self._logits_nd)

    def optimizer(self):
        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate=self._rl)
        train_op = optimizer.minimize(loss=self._loss_nd, global_step=tf.train.get_global_step())
        self._train_optimizer_nd = train_op

    def prediction(self):
        # PREDICTION
        self._pred_nd = tf.argmax(self._logits_nd, 1)

    def accuracy(self):
        # ACCURACY
        self._acc_nd = tf.equal(self._pred_nd, tf.argmax(cnn._label_nd, 1))

cnn = CNN()
    
cnn.architecture(cnn._feature_nd)
cnn.cost_function()
cnn.optimizer()
cnn.prediction()
cnn.accuracy()

plt.ion()
plt.close('all')
fig_pred, ax_pred = plt.subplots(5, 5, figsize=(10, 10))

plt.show()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in tqdm(range(cnn._epoch)):
        batch_index = np.random.choice(len(cnn._train_data), size=cnn._batch_size)
        batch_x = cnn._train_data[batch_index]
        batch_y = cnn._train_labels[batch_index]

        sess.run(cnn._train_optimizer_nd, feed_dict={cnn._feature_nd: batch_x, cnn._label_nd: batch_y})

        if iter % 100 == 0:
            pred1 = sess.run(cnn._pred_nd, feed_dict={cnn._feature_nd: cnn._test_data})
            pred2 = sess.run(cnn._pred_nd, feed_dict={cnn._feature_nd: batch_x})

            train_acc = sess.run(cnn._acc_nd, feed_dict={cnn._feature_nd: cnn._train_data[:cnn._batch_size], cnn._label_nd: cnn._train_labels[:cnn._batch_size]})
            eval_acc = sess.run(cnn._acc_nd, feed_dict={cnn._feature_nd: cnn._eval_data[:cnn._batch_size], cnn._label_nd: cnn._eval_labels[:cnn._batch_size]})
            
            print("({0}) TRAIN ACC.: {1} % | EVAL ACC.: {2} %"\
                  .format(iter, np.sum(train_acc)/len(train_acc)*100, np.sum(eval_acc)/len(eval_acc)*100))

            #print(pred[:10])
            
            for i in range(5):
                for j in range(5):
                    if (i == 0) and (j==0):
                        ax_pred[i, j].imshow(np.asarray(cnn._test_data[i*5+j], dtype=np.float32))

                        if pred1[i*5+j] == 0:
                            ax_pred[i, j].set_title("Face")
                        else:
                            ax_pred[i, j].set_title("Non")
                    else:
                        ax_pred[i, j].imshow(np.asarray(batch_x[i*5+j], dtype=np.float32))
                        #print("===> ", pred[i*5+j])
                        if pred2[i*5+j] == 0:
                            ax_pred[i, j].set_title("Face")
                        else:
                            ax_pred[i, j].set_title("Non")

            plt.tight_layout()
            plt.pause(0.0001)
            plt.draw()

'''
if __name__ == "__main__":
    cnn_fn(train_data, train_labels, None)
'''
