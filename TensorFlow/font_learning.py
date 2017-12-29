import tensorflow as tf
import numpy as np
from pickle_helper import PickleHelper

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_set, valid_set, test_set = PickleHelper.load_pickle(path = "../Data/", name = "mnist.pkl")

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e


def data_processing(dataset):
    
    x = np.array([np.reshape(x, 784) for x in dataset[0]])
    y = np.array([vectorized_result(y) for y in dataset[1]])
    
    return x, y

def mini_batch(data_x, data_y, batch_size = 10):
    num_trainset = len(data_x) # get a number of samples in dataset

    random_mask = np.random.choice(num_trainset, batch_size)

    batch_x = data_x[random_mask]
    batch_y = data_y[random_mask]

    return batch_x, batch_y


train_x, train_y = data_processing(train_set)
print type(train_x)
test_x, test_y = data_processing(test_set)


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

l = tf.matmul(x, W) + b
y = tf.nn.softmax(l)

cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = mini_batch(train_x, train_y, batch_size = 50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 50 == 0:
        #print np.shape(batch_xs), np.shape(batch_ys)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print sess.run(accuracy, feed_dict={x: test_x, y_: test_y})