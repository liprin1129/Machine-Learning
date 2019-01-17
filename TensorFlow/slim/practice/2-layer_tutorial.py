''' # Native TF: layer

with tf.variable_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1, name='weight'))
    conv = tf.nn.conv2d(input=input_val, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name='activation')
'''

''' # Slim-TF: layer
# padding='SAME' is default
# strindes=[1,1,1,1] is default
net = slim.conv2d(inputs=input_val, num_outputs=128, kernel_size=[3,3], scope='conv1_1')
'''

######################################
# Repeat and Stack: same output size #
######################################
''' # Native TF: repeat option 1
net1 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.variable_scope('test1') as scope:
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_1')
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_2')
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_3')
  net1 = slim.max_pool2d(net1, [2,2], scope='pool2')
'''

''' # Native TF: repeat option 2
# use for loop
net2 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.variable_scope('test2') as scope:
  for i in range(3):
    net2 = slim.conv2d(net2, 256, [3,3], scope='conv3_%d' % (i+1))
  net2 = slim.max_pool2d(net2, [2,2], scope='pool2')
'''

''' # TF-Slim: repeat
net3 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.variable_scope('test3') as scope:
  net3 = slim.repeat(net3, 3, slim.conv2d, 256, [3,3], scope='conv3')
  net3 = slim.max_pool2d(net2, [2,2], scope='pool2')
'''


###########################################
# Repeat and Stack: different output size #
###########################################
''' # Native TF: repeat fully connected layers of different output
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 4])
  mlp1 = slim.fully_connected(inputs=input_val, num_outputs=32, scope='fc/fc_1')
  mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=64, scope='fc/fc_2')
  mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=128, scope='fc/fc_3')
'''

''' # TF-Slim: stack fully connected layers of different output
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 4])
  mlp2 = slim.stack(input_val, slim.fully_connected, [32, 64, 128], scope='fc')
'''

''' # Native TF: repeat convolutional layers of different output
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 32, 32, 8])
  conv1 = slim.conv2d(input_val, 32, [3,3], scope='core/core_1')
  conv1 = slim.conv2d(conv1, 32, [1, 1], scope='core/core_2')
  conv1 = slim.conv2d(conv1, 64, [3, 3], scope='core/core_3')
  conv1 = slim.conv2d(conv1, 64, [1, 1], scope='core/core_4')
'''

''' # TF-Slim: stack covolutional layers of different output
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 32, 32, 8])
  conv2 = slim.stack(input_val, slim.conv2d, [(32,[3,3]), (32,[1,1]), (64,[3,3]), (64,[1,1])], scope='core')
'''