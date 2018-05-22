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

        self.stride = 1


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

    def simple_CNN(self, input_node):
        # CONVOLUTIONAL 1
        conv1 = self._convolutional_layer(input_node, 3, 32, 1, "VALID", "conv1")
        
        # FLATTEN
        relu1_shape = conv1.get_shape().as_list()
        conv1_flat = tf.reshape(conv1, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])
        
        # DENSE 1
        dense1 = self._dense_layer(conv1_flat, 128, "relu", "dense1")

        # DENSE 2
        dense2 = self._dense_layer(dense1, 2, "None", "dense2")
        
        self._logits_nd = dense2

    def VGG16(self, input_node):
        conv1_1 = self._convolutional_layer(input_node, self.kernel_3x3, self.depth1, self.stride, "SAME", "conv1_1")
        conv1_2 = self._convolutional_layer(conv1_1, self.kernel_3x3, self.depth1, self.stride, "SAME", "conv1_2")
        max_pool1 = self._max_pool(conv1_2, "max_pool1")

        conv2_1 = self._convolutional_layer(max_pool1, self.kernel_3x3, self.depth2, self.stride, "SAME", "conv2_1")
        conv2_2 = self._convolutional_layer(conv2_1, self.kernel_3x3, self.depth2, self.stride, "SAME", "conv2_2")
        max_pool2 = self._max_pool(conv2_2, "max_pool2")

        conv3_1 = self._convolutional_layer(max_pool2, self.kernel_3x3, self.depth3, self.stride, "SAME", "conv3_1")
        conv3_2 = self._convolutional_layer(conv3_1, self.kernel_3x3, self.depth3, self.stride, "SAME", "conv3_2")
        conv3_3 = self._convolutional_layer(conv3_2, self.kernel_1x1, self.depth3, self.stride, "SAME", "conv3_3")
        max_pool3 = self._max_pool(conv3_3, "max_pool3")

        conv4_1 = self._convolutional_layer(max_pool3, self.kernel_3x3, self.depth4, self.stride, "SAME", "conv4_1")
        conv4_2 = self._convolutional_layer(conv4_1, self.kernel_3x3, self.depth4, self.stride, "SAME", "conv4_2")
        conv4_3 = self._convolutional_layer(conv4_2, self.kernel_1x1, self.depth4, self.stride, "SAME", "conv4_3")
        max_pool4 = self._max_pool(conv4_3, "max_pool4")

        conv5_1 = self._convolutional_layer(max_pool4, self.kernel_3x3, self.depth5, self.stride, "SAME", "conv5_1")
        conv5_2 = self._convolutional_layer(conv2_1, self.kernel_3x3, self.depth5, self.stride, "SAME", "conv5_2")
        conv5_3 = self._convolutional_layer(conv2_2, self.kernel_1x1, self.depth5, self.stride, "SAME", "conv5_3")
        max_pool5 = self._max_pool(conv5_3, "max_pool5")

        # FLATTEN
        relu1_shape = max_pool5.get_shape().as_list()
        max_pool5_flat = tf.reshape(max_pool5, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])
        
        #print(max_pool5_flatten.get_shape())
        dense1 = self._dense_layer(max_pool5_flat, self.dense_4096, "relu", name="dense1")
        dense2 = self._dense_layer(dense1, self.dense_4096, "relu", name="dense2")
        dense3 = self._dense_layer(dense2, self.dense_1000, "relu", name="dense3")
        dense_output = self._dense_layer(dense3, self.dense_output, None, name="dense_output")
        #dense_output = tf.nn.dropout(dense_output, keep_prob_node)
        # Dens -> Probabilities
        #probabilities = tf.nn.softmax(dense3, name="probabilities")
        #self.__architecture = probabilities

        # Dens -> Logits
        #logits = tf.nn.relu(dense_output)
        self._logits_nd = dense_output

    def _cost_function(self):
        # CONST FUNCTION
        self._loss_nd = tf.losses.softmax_cross_entropy(onehot_labels=self._label_nd, logits=self._logits_nd)

    def _optimizer(self):
        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate=self._rl)
        train_op = optimizer.minimize(loss=self._loss_nd, global_step=tf.train.get_global_step())
        self._train_optimizer_nd = train_op

    def _prediction(self):
        # PREDICTION
        self._pred_nd = tf.argmax(self._logits_nd, 1)

    def _accuracy(self):
        # ACCURACY
        self._acc_nd = tf.equal(self._pred_nd, tf.argmax(cnn._label_nd, 1))

    def net_initializer(self):
        #self.simple_CNN(self._feature_nd)
        self.VGG16(self._feature_nd)
        self._cost_function()
        self._optimizer()
        self._prediction()
        self._accuracy()

    def train(self):
        with tf.Session() as sess:
            self.net_initializer()

            sess.run(tf.global_variables_initializer())

            for iter in tqdm(range(cnn._epoch)):
                batch_index = np.random.choice(len(cnn._train_data), size=cnn._batch_size)
                batch_x = cnn._train_data[batch_index]
                batch_y = cnn._train_labels[batch_index]

                sess.run(cnn._train_optimizer_nd, feed_dict={cnn._feature_nd: batch_x, cnn._label_nd: batch_y})

                if iter % 100 == 0:
                    self._train_info(sess)

    def _train_info(self, session):
        #pred1 = session.run(cnn._pred_nd, feed_dict={cnn._feature_nd: cnn._test_data})
        #pred2 = session.run(cnn._pred_nd, feed_dict={cnn._feature_nd: batch_x})

        train_acc = session.run(cnn._acc_nd, feed_dict={cnn._feature_nd: cnn._train_data[:cnn._batch_size], cnn._label_nd: cnn._train_labels[:cnn._batch_size]})
        eval_acc = session.run(cnn._acc_nd, feed_dict={cnn._feature_nd: cnn._eval_data[:cnn._batch_size], cnn._label_nd: cnn._eval_labels[:cnn._batch_size]})

        print("({0}) TRAIN ACC.: {1} % | EVAL ACC.: {2} %"\
              .format(iter, np.sum(train_acc)/len(train_acc)*100, np.sum(eval_acc)/len(eval_acc)*100))

'''
plt.ion()
plt.close('all')
fig_pred, ax_pred = plt.subplots(5, 5, figsize=(10, 10))

plt.show()
'''

cnn = CNN()
cnn.net_initializer()
cnn.train()

'''
with tf.Session() as sess:
    cnn = CNN()
    cnn.net_initializer()
    
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

            #print(pred2[:10])
            #print(sess.run(cnn._logits_nd, feed_dict={cnn._feature_nd: batch_x}))
            #print(cnn._train_labels[:cnn._batch_size][:10])

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

'''
if __name__ == "__main__":
    cnn_fn(train_data, train_labels, None)
'''
