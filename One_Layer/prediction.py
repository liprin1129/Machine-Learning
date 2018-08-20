from pickle_helper import PickleHelper
from network import Network
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

train_set, valid_set, test_set = PickleHelper.load_pickle(path = "../Data/", name = "mnist.pkl")
W, B = PickleHelper.load_pickle(path="./", name="minist_one_layer.pkl")

train_x = [np.reshape(x, (784, 1)) for x in train_set[0]]
train_x =np.array(train_x)
test_x = [np.reshape(x, (784, 1)) for x in test_set[0]]
test_x = np.array(test_x)

ann = Network([784, 30, 10])
ann.weights = W
ann.biases = B
print np.argmax(ann.feedforward(train_x[10]))

fig, ax = plt.subplots(1, 5, figsize=(10, 2))
#plt.show()

'''
indice = np.random.choice(len(test_x), size=5)
print indice
test_x = np.array(test_x)
test_x[indice]
'''

for iter in tqdm(range(100)):

    indice = np.random.choice(len(test_x), size=5)
    
    input_data = test_x[indice]

    for i in range(5):
        pred = np.argmax(ann.feedforward(input_data[i]))
        ax[i].imshow(np.reshape(input_data[i], (28, 28)), cmap="gray")
        ax[i].set_title(pred)

    #plt.suptitle("Prediction")
    plt.tight_layout()
    plt.pause(1)
    plt.draw()
