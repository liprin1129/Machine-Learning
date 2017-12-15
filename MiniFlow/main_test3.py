from input import Input
from network import Network
from linear import Linear
from activation import Sigmoid
from cost import L2
from update import Update
import numpy as np

# One layer (two input): 3 -> 2

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()
W3, b3 = Input(), Input()

# Training dataset
X_ = np.reshape(np.array([[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]), (2, 3))
W1_ = np.random.randn(3, 10)
b1_ = np.random.randn(10)
W2_ = np.random.randn(10, 2)
b2_ = np.random.randn(2)
y_ = np.array([[1., 0.], [0., 1.]])

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
cost = L2(y, s2)

feed_dict = {X: X_, y: y_,
             W1: W1_, b1: b1_,
             W2: W2_, b2: b2_}

hyper_parameters = [W1, b1, W2, b2]

graph = Network.topological_sort(feed_dict)

epoch = 1000000
for i in xrange(epoch):
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
    Update.stochastic_gradient_descent(hyper_parameters, learning_rate=3e-2)

    if cost.value < 1e-20:
        break

    if i % 10000 == 0:
        print "<EPOCH : {0}>\n\n".format(i)
        print "COST: ", cost.value, "\nPRED Y:\n", np.round(cost.pred_y, 3), "\n"
        """
        print "W1: ", np.shape(W1.value)
        print "b1: ", np.shape(b1.value)#, "\n", b1.value
        print "W2: ", np.shape(W2.value)
        print "b2: ", np.shape(b2.value)#, "\n", b2.value
        """

        # Evaluation #
        """
        # Test dataset
        X_t, y_t = Input(), Input()
        W1_t, b1_t = Input(), Input()
        W2_t, b2_t = Input(), Input()
        W3_t, b3_t = Input(), Input()

        l1_t = Linear(X_t, W1_t, b1_t)
        s1_t = Sigmoid(l1_t)
        l2_t = Linear(s1_t, W2_t, b2_t)
        s2_t = Sigmoid(l2_t)
        cost_t = L2(y_t, s2_t)

        X_t_ = np.reshape(np.array([0.3, 0.8, 0.3]), (1, 3))
        y_t_ = np.array([[1., 0.]])
        
        feed_dict_t = {X_t: X_t_, y_t: y_t_,
                       W1_t: W1.value, b1_t: b1.value,
                       W2_t: W2.value, b2_t: b2.value}
        
        graph_t = Network.topological_sort(feed_dict_t)
        
        Network.forward_propagation(graph_t)
        """
        
        X_t_ = np.reshape(np.array([0.3, 0.8, 0.3]), (1, 3))
        y_t_ = np.array([[1., 0.]])

        Network.evaluate(graph, hyper_parameters, [X, y], [X_t_, y_t_])
