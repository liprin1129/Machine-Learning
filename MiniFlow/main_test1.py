from input import Input
from network import Network
from linear import Linear
from activation import Sigmoid
from cost import L2
from update import Update
import numpy as np

# One layer (one input): 2 -> 1
X, y = Input(), Input()
W, b = Input(), Input()

# Training dataset
X_ = np.reshape(np.array([0.1, 0.2]), (1, 2))
W_ = np.reshape(np.array([1., 2.]), (2, 1))
b_ = np.reshape(np.array([0.]), (-1, 1))
y_ = np.reshape(np.array([1.]), (-1, 1))

# Test dataset
X_test_ = np.reshape(np.array([0.1, 0.1]), (1, 2))
X_test = Input()
X_test.value = X_test_

l1 = Linear(X, W, b)
s1 = Sigmoid(l1)
cost = L2(y, l1)

feed_dict = {X: X_, y: y_,
             W: W_, b: b_}

hyper_parameters = [W, b]

graph = Network.topological_sort(feed_dict)
#test_graph = [X_test, W, b] # Sorted nodes for evaluation

epoch = 1000000
for i in xrange(epoch):
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
    Update.stochastic_gradient_descent(hyper_parameters, learning_rate=1e-5)

    if i % 10000 == 0:
        print "<EPOCH : {0}>\n".format(i), "COST: ", cost.value, "\nPRED Y:\n", np.round(cost.pred_y), "\n"
        Network.evaluate(X_test, graph)
        #Network.forward_propagation(test_graph)
        #print cost.pred_y
    
    if cost.value < 1e-20:
        break
