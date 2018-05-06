'''
import platform
if platform.system() is "Darwin":
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
else:
        import matplotlib.pyplot as plt
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import flatten
from tensorflow.python.framework import ops

# START A GRAPH SESSION
sess = tf.Session()

# DOWNLOAD DATA
data_dir = '../Data/HandWritten_Data/'
mnist = read_data_sets(data_dir)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
#train_xdata = np.expand_dims(train_xdata, axis=3)

test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
#test_xdata = np.expand_dims(test_xdata, axis=3)

# CONVERT LABELS INTO ONE-HOT ENCODED VECTORS
train_labels = mnist.train.labels
test_labels = mnist.test.labels

#train_labels = tf.one_hot(train_labels, 10)
#test_labels = tf.one_hot(test_labels, 10)
#print(train_labels[0])

# SET MODEL PARAMETERS
batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1 # greyscale = 1 channel
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

# MODEL PLACEHOLDER
x_input_shape = (None, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(None))

eval_input_shape = (None, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape=(None))

# CONVOLUTIONAL LAYER VARIABLES
#conv1_W = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],
conv1_W = tf.Variable(tf.random_normal([4, 4, num_channels, conv1_features],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv1_b = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

#conv2_W = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],
conv2_W = tf.Variable(tf.random_normal([4, 4, conv1_features, conv2_features],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv2_b = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

#fully1_W = tf.Variable(tf.truncated_normal([4*4*50, fully_connected_size1], stddev=0.1, dtype=tf.float32))
fully1_W = tf.Variable(tf.random_normal([4*4*50, fully_connected_size1], stddev=0.1, dtype=tf.float32))
fully1_b = tf.Variable(tf.random_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
#fully1_b = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

#fully2_W = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
fully2_W = tf.Variable(tf.random_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
#fully2_b = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
fully2_b = tf.Variable(tf.random_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_conv_net(input_data):
    # FIRST CONV -> RELU -> MAXPOOL
    conv1 = tf.nn.conv2d(input_data, conv1_W, strides=[1,1,1,1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='VALID')

    # FIRST CONV -> RELU -> MAXPOOL
    conv2 = tf.nn.conv2d(max_pool1, conv2_W, strides=[1,1,1,1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides = [1, max_pool_size2, max_pool_size2, 1], padding='VALID')

    # FLATTEN
    flatten_node = flatten(max_pool2)


    # FIRST FULLY
    fully_connected_1 = tf.nn.relu(tf.add(tf.matmul(flatten_node, fully1_W), fully1_b))

    # SECOND FULLY
    final_model_output = tf.add(tf.matmul(fully_connected_1, fully2_W), fully2_b)

    return(final_model_output)
    
model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

# LOSS FUNCTION (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

# PREDICTION FUNCTION
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

# ACCURACY FUNCTION
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return(100. * num_correct/batch_predictions.shape[0])

# OPTIMIZER
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# INITIALIZE
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess.run(init)

# TRAIN START
train_loss = []
train_acc = []
test_acc = []

'''
train_xdata = np.expand_dims(train_xdata, axis=3)
a = sess.run(loss, feed_dict={x_input: train_xdata, y_target: train_labels})
print(a)
'''

for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, axis=3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}

    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if (i+1) % eval_every == 0:
        eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)
        
        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

'''
#a = sess.run(model_output, feed_dict={x_input:train_xdata, y_target:train_labels})
#a = sess.run(prediction, feed_dict={x_input:train_xdata, y_target:train_labels})
a = sess.run(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target),
             feed_dict={x_input:train_xdata, y_target:train_labels})
print(a[0], a[100], train_labels[0], train_labels[100])

print(train_labels.shape)
'''
