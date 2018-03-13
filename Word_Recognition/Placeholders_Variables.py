import tensorflow as tf

# MODEL PLACEHOLDER
x_input_shape = (None, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(None))

eval_input_shape = (None, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape=(None))

# CONVOLUTIONAL LAYER VARIABLES
conv1_W = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_depth],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv1_b = tf.Variable(tf.zeros([conv1_depth], dtype=tf.float32))

conv2_W = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_depth],
                                          stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
conv2_b = tf.Variable(tf.zeros([conv2_depth], dtype=tf.float32))

fully1_W = tf.Variable(tf.truncated_normal([4*4*50, fully_connected_size1], stddev=0.1, dtype=tf.float32))
fully1_b = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

fully2_W = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
fully2_b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1, dtype=tf.float32))
