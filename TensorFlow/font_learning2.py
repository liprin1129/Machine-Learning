import tensorflow as tf
import numpy as np
from pickle_helper import PickleHelper

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_set = PickleHelper.load_pickle(path = "../Data/", name = "HGRGM.pkl")
test_set = PickleHelper.load_pickle(path = "../Data/", name = "HGRGE.pkl")
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

    #print(random_mask)
    batch_x = data_x[random_mask]
    batch_y = data_y[random_mask]

    return batch_x, batch_y


train_x = np.array(train_set[0])
matrix_y = train_set[1]
train_y = []

for i in matrix_y:
    train_y.append(np.argmax(i))
    #print(i)

train_y = np.array(train_y)
train_y = train_y.reshape(-1, 1)
#print(train_y.shape)
batch_xs, batch_ys = mini_batch(train_x, train_y, batch_size = 50)

#print( batch_xs.shape, batch_ys.shape)

test_x = test_set[0]
test_y = test_set[1]

x = tf.placeholder(tf.float32, [None, 784])
#y_ = tf.placeholder(tf.float32, [None, 2136])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))

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
        sess.run(y, feed_dict={x: train_x[0]})
        '''
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: train_x[100:200], y_: train_y[100:200]}))
        '''