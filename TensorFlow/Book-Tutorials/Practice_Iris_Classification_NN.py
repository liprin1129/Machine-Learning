import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# LOAD IRIS DATA
iris = datasets.load_iris()
binary_target = iris.target
iris_2d = iris.data


