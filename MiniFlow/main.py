from input import Input
from network import Network
from linear import Linear
from activation import Sigmoid
from cost import L2
from update import Update
import numpy as np
import os, sys
from pickle_helper import PickleHelper
from tqdm import tqdm
from random import shuffle

"""
global_path = os.path.abspath("../data/")

if global_path not in sys.path:
    sys.path.append(global_path)
"""

train_set, valid_set, test_set = PickleHelper.load_pickle(path = "../Data/", name = "mnist.pkl")

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e


def data_processing(train_set):
    
    train_x = np.array([np.reshape(x, 784) for x in train_set[0]])
    train_y = np.array([vectorized_result(y) for y in train_set[1]])
        
    return train_x, train_y

test_x = [np.reshape(x, 784) for x in test_set[0]]
test_y = np.array([vectorized_result(y) for y in test_set[1]])

"""
print "X: ", np.shape(train_x)
print "Y: ", np.shape(train_y)
print "X TEST: ", np.shape(test_x)
print "Y TEST: ", np.shape(test_y)
"""

# ######## #
# Training #
# ######## #

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

W1_ = np.random.randn(784, 30)
b1_ = np.random.randn(30)
W2_ = np.random.randn(30, 10)
b2_ = np.random.randn(10)

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
cost = L2(y, s2)

feed_dict = {X: train_x, y: train_y,
             W1: W1_, b1: b1_,
             W2: W2_, b2: b2_}

hyper_parameters = [W1, b1, W2, b2]

graph = Network.topological_sort(feed_dict)

epoch = 30
for i in tqdm(xrange(epoch)):

    shuffle(training_data)

    mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]

    for mini_batch in mini_batches:

        
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
    Update.stochastic_gradient_descent(hyper_parameters, learning_rate = 1e-3)

    if cost.value < 1e-5:
        break

    if i % 10 == 0:
        print "<EPOCH : {0}>\n".format(i), "COST: ", cost.value

        Network.evaluate(graph, hyper_parameters, [X, y], [test_x, test_y])
