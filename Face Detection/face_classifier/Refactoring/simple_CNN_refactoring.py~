import tensorflow as tf
from data_wrangling import PickleHelper
from tqdm import tqdm
import numpy as np

import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

class net_setup(object):
    def __init__(self):
        self.epoch = 5000
        self.batch_size = 500
        self.rl = 0.001 #learning rate
        
        self.data = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-features-norm.pkl")
        self.labels = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-labels-norm.pkl")
        self.test_data = PickleHelper.load_pickle("../../Data/Face/", "blob-itamochi-32x32.pkl")
        
        print("DATA: ", self.data.shape)
        
        # FOR CODE TESTING
        self.train_data = self.data[:100]
        self.train_labels = self.labels[:100]
        self.eval_data = self.data[100:120]
        self.eval_labels = self.labels[100:120]
        
        # CODE IMPLEMENTATION
        self.train_data = self.data[:80000]
        self.train_labels = self.labels[:80000]
        #eval_data = data[80000:]
        #eval_labels = labels[80000:]

class CNN(net_setup):
    def __init__(self):
        super(CNN, self).__init__()

        self.feature_node = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.label_node = tf.placeholder(tf.float32, shape=[None, 2])
        
        self.img_channel = 3
        
        self.logits = None
        self.loss = None
        self.train_optimizer = None
        self.pred = None
        self.acc = None
        
    def architecture(self, input_node):
        # CONVOLUTIONAL 1
        conv1_W = tf.Variable(
            tf.truncated_normal([3, 3, 3, 32], stddev=0.1, dtype=tf.float32))
        conv1_b = tf.Variable(tf.truncated_normal([32], stddev=0.1, dtype=tf.float32))
        
        conv1 = tf.layers.conv2d(inputs=input_node, filters=32, kernel_size=[3, 3], strides=(1, 1),  padding="valid", activation=tf.nn.relu)
    
        #conv1 = tf.nn.conv2d(feature_node, conv1_W, strides=[1, 1, 1, 1], padding='VALID')
        #relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
        #pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)
        
        # FLATTEN
        relu1_shape = conv1.get_shape().as_list()
        conv1_flat = tf.reshape(conv1, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])
        
        # DENSE 1
        dense1 = tf.layers.dense(inputs=conv1_flat, units=128, activation=tf.nn.relu)#tf.add(tf.matmul(
        
        # DENSE 2
        dense2 = tf.layers.dense(inputs=dense1, units=2)#tf.add(tf.matmul(
        
        self.logits = dense2

    def cost_function(self):
        # CONST FUNCTION
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label_node, logits=self.logits)

    def optimizer(self):
        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate=self.rl)
        train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())
        self.train_optimizer = train_op

    def prediction(self):
        # PREDICTION
        self.pred = tf.argmax(self.logits, 1)

    def accuracy(self):
        # ACCURACY
        self.acc = tf.equal(self.pred, tf.argmax(cnn.label_node, 1))

cnn = CNN()
    
cnn.architecture(cnn.feature_node)
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

    for iter in tqdm(range(cnn.epoch)):
        batch_index = np.random.choice(len(cnn.train_data), size=cnn.batch_size)
        batch_x = cnn.train_data[batch_index]
        batch_y = cnn.train_labels[batch_index]

        sess.run(cnn.train_optimizer, feed_dict={cnn.feature_node: batch_x, cnn.label_node: batch_y})

        if iter % 100 == 0:
            pred1 = sess.run(cnn.pred, feed_dict={cnn.feature_node: cnn.test_data})
            pred2 = sess.run(cnn.pred, feed_dict={cnn.feature_node: batch_x})

            train_acc = sess.run(cnn.acc, feed_dict={cnn.feature_node: cnn.train_data[:cnn.batch_size], cnn.label_node: cnn.train_labels[:cnn.batch_size]})
            eval_acc = sess.run(cnn.acc, feed_dict={cnn.feature_node: cnn.eval_data[:cnn.batch_size], cnn.label_node: cnn.eval_labels[:cnn.batch_size]})
            
            print("({0}) TRAIN ACC.: {1} % | EVAL ACC.: {2} %"\
                  .format(iter, np.sum(train_acc)/len(train_acc)*100, np.sum(eval_acc)/len(eval_acc)*100))

            #print(pred[:10])
            
            for i in range(5):
                for j in range(5):
                    if (i == 0) and (j==0):
                        ax_pred[i, j].imshow(cnn.test_data[i*5+j])

                        if pred1[i*5+j] == 0:
                            ax_pred[i, j].set_title("Face")
                        else:
                            ax_pred[i, j].set_title("Non")
                    else:
                        ax_pred[i, j].imshow(batch_x[i*5+j])
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
