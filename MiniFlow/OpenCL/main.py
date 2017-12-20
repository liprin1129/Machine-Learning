from input import Input
from network import Network
from linear import Linear
from activation import Sigmoid
from cost import SSE
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
test_x, test_y = data_processing(test_set)

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
cost = SSE(y, s2)

feed_dict = {X: train_x, y: train_y,
             W1: W1_, b1: b1_,
             W2: W2_, b2: b2_}

hyper_parameters = [W1, b1, W2, b2]

graph = Network.topological_sort(feed_dict)

epoch = 1000
batch_size = 30
steps_per_batch = len(train_y) // batch_size

for i in tqdm(xrange(epoch)):
    for j in xrange(steps_per_batch):
        
        batch_x, batch_y = mini_batch(train_x, train_y)

        X.value = batch_x
        y.value = batch_y
        
        Network.forward_propagation(graph)
        Network.backward_propagation(graph)
        Update.stochastic_gradient_descent(hyper_parameters, learning_rate = 3e-3)

        if cost.value < 1e-5:
            break

    if i % 10 == 0:
        test_result = Network.evaluate(graph, hyper_parameters, [X, y], [test_x, test_y])
        print "<EPOCH : {0}>\n".format(i), "TRAIN COST: {0}, TEST COST: {1}%".format(cost.value, test_result)

PickleHelper.save_to_pickle(path = "./", name = "trained_data.p", data = [graph, hyper_parameters])
