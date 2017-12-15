from input import Input
from network import Network
from linear import Linear
from activation import Sigmoid
from cost import L2
from update import Update
import numpy as np


# Two layer (two frequency): 3 -> 2 -> 1
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

X_ = np.reshape(np.array([[-1., -2., -3.], [1., 2., 3.]]), (2, 3))
W1_ = np.random.randn(3, 2)
b1_ = np.random.randn(2, 2)
W2_ = np.random.randn(2, 1)
b2_ = np.random.randn(2, 1)
y_ = np.reshape(np.array([[1.], [0.]]), (-1, 1), (-1, 1))

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = L2(y, l2)

feed_dict = {X: X_, y: y_,
             W1: W1_, b1: b1_,
             W2: W2_, b2: b2_}
hyper_parameters = [W1, b1, W2, b2]

graph = Network.topological_sort(feed_dict)

epoch = 1000000
for i in xrange(epoch):
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
    Update.stochastic_gradient_descent(hyper_parameters, learning_rate = 1e-4)

    if cost.value < 1e-20:
        break

    if i % 10000 == 0:
        print "<EPOCH : {0}>\n".format(i), "COST: ", cost.value, "\nPRED Y:\n", np.round(cost.pred_y, 3), "\n"
