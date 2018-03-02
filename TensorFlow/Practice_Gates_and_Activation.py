import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

batch_size = 50

a1 = tf.Variable(tf.random_normal(shape=[1,1]))

print(sess.run(tf.random_normal(shape=[1,1])))
