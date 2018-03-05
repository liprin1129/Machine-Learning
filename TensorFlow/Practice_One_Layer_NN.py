import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# COLLECTING DATA
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

# CREATE GRAPH SESSION
sess = tf.Session()

# MAKE RESULTS REPRODUCIBLE
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

## USING SKLEARN PACKAGE
'''
### TRAIN AND TEST SET SPLITTING
from sklearn.model_selection import train_test_split
x_vals_train, x_vals_test, y_vals_train, t_vals_test =train_test_split(x_vals, y_vals, train_size = 0.8, random_state = seed)

### NOMALISATION BY COLUMN
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_vals_train)
x_vals_train = scaler.transform(x_vals_train)

scaler.fit(x_vals_test)
x_vals_test = scaler.transform(x_vals_test)
'''

## USING OWN METHOD
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

### NORMALIZATION BY COLUMN
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

'''
print('ORIGINAL DATA SHAPE: ', x_vals.shape, y_vals.shape)
print('TRAIN & TEST SHAPE: ', x_vals_train.shape, x_vals_test.shape, y_vals_train.shape, y_vals_test.shape)

for i in range(3):
    print(np.max(x_vals_train[:, i]), np.min(x_vals_train[:, i]))
'''

# BATCH SIZE
batch_size = 50

# VARIABLES AND PLACEHOLDER
hidden_layer_nodes = 10
W1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
W2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# MODEL
'''
l1_matmul = tf.matmul(x_data, W1)
l1_add = tf.add(l1_matmul, b1)
l1_relu = tf.nn.relu(l1_add)

o_matmul = tf.matmul(l1_relu, W2)
o_add = tf.add(o_matmul, b2)
o_relu = tf.nn.relu(o_add)
'''
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
o_relu = tf.nn.relu(tf.add(tf.matmul(hidden_output, W2), b2))

# COST FUNCTION
loss = tf.reduce_mean(tf.square(y_target - o_relu))

# OPTIMIZER
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# INITIALIZE
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
