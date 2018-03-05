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
b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
a2 = tf.Variable(tf.random_normal(shape=[1,1]))
b2 = tf.Variable(tf.random_uniform(shape=[1,1]))

x = np.random.normal(2, 0.1, 500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

matmul_sigmoid = tf.matmul(x_data, a1)
add_sigmoid = tf.add(matmul_sigmoid, b1)
sigmoid_activation = tf.sigmoid(add_sigmoid)

matmul_relu = tf.matmul(x_data, a2)
add_relu = tf.add(matmul_relu, b2)
relu_activation = tf.nn.relu(add_relu)

# DECLARE THE LOSS FUNCTION AS THE DIFFERENCE BETWEEN
# THE OUTPUT AND A TARGET VALUE, 0.75
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

# INITIALIZE VARIABLES
init = tf.global_variables_initializer()
sess.run(init)

# DECLARE OPTIMIZER
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu = my_opt.minimize(loss2)

# RUN LOOP ACROSS GATE
print('\nOptimizing Sigmoid AND Relu Output to 0.75')

loss_vec_sigmoid = []
loss_vec_relu = []

for i in range(500):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])

    #print(np.shape(x[rand_indices]), np.shape(x_vals))
    sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
    sess.run(train_step_relu, feed_dict={x_data: x_vals})

    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))

    sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals}))
    relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals}))

    if i%50==0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))
        #print('sigmoid = ', np.shape(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})), ' relu = ' + str(np.mean(relu_output)))

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
