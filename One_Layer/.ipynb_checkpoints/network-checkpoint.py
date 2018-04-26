import os
import sys
import numpy as np
from random import shuffle

global_path = os.path.abspath("../image_to_text")

if global_path not in sys.path:
    sys.path.append(global_path)

from pickle_helper import PickleHelper

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        print("\nNetwork is generated >>\n")
        for idx, (weight, biase) in enumerate(zip(self.weights, self.biases)):
            print("Weights {0}: {1}\nBiases {0}: {2}\n".format(
                idx, np.shape(weight), np.shape(biase)))
        
    def feedforward(self, a):
        # c = 0
        for b, w in zip(self.biases, self.weights):
            # print("Count ", c)
            # c += 1
            # print("a: {0}, w: {1}\n".format(np.shape(a), np.shape(w)))
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs = 30,
            mini_batch_size = 10, eta = 3.0, test_data = None):

        n = len(training_data)
        if test_data:
            n_test = len(test_data)

        for j in xrange(epochs):
            shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}%".format(
                    j, self.evaluate(test_data)/float(n_test) * 100))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

if __name__ == "__main__":

    train_set, valid_set, test_set = PickleHelper.load_pickle(path = "./", name = "mnist.pkl")

    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
    
    train_x = [np.reshape(x, (784, 1)) for x in train_set[0]]
    train_y = [vectorized_result(y) for y in train_set[1]]
    
    test_x = [np.reshape(x, (784, 1)) for x in test_set[0]]

    #print(np.shape(train_x), np.shape(train_y))
    #print(np.max(train_x[0]))

    import matplotlib.pyplot as plt

    plt.imshow(np.reshape(train_x[10], (28, 28)), cmap="gray")
    plt.title(train_y[10].T)
    plt.show()

    ann = Network([784, 30, 10])
    ann.SGD(zip(train_x, train_y), test_data = zip(test_x, test_set[1]))
