import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import flatten
from tensorflow.python.framework import ops

def font_CNN_model(input_data,
                   conv1_W, conv1_b, max_pool_size1,
                   conv2_W, conv2_b, max_pool_size2,
                   fully1_W, fully1_b,
                   fully2_W, fully2_b):
    # MODEL
    ## FIRST CONV -> RELU -> MAXPOOL
    conv1 = tf.nn.conv2d(input_data, conv1_W, strides=[1,1,1,1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='VALID')

    ## FIRST CONV -> RELU -> MAXPOOL
    conv2 = tf.nn.conv2d(max_pool1, conv2_W, strides=[1,1,1,1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides = [1, max_pool_size2, max_pool_size2, 1], padding='VALID')

    ## FLATTEN & Fully Connected Layer Variable
    flatten_node = flatten(max_pool2)
    print("Flatten_shape: ", flatten_node.get_shape())
    
    ## FIRST FULLY
    fully_connected_1 = tf.nn.relu(tf.add(tf.matmul(flatten_node, fully1_W), fully1_b))

    ## SECOND FULLY
    final_model_output = tf.add(tf.matmul(fully_connected_1, fully2_W), fully2_b)

    return(final_model_output)

# ACCURACY FUNCTION
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    #num_correct = np.sum(np.equal(batch_predictions, targets))
    #return(100. * num_correct/batch_predictions.shape[0])
    return batch_predictions

def font_recognition(train_xdata, train_labels, test_xdata, test_labels, learning_rate,
                     conv1_filter, conv1_depth, max_pool_size1,
                     conv2_filter, conv2_depth, max_pool_size2,
                     fully_size1, target_size):
    
    # START A GRAPH SESSION
    sess = tf.Session()

    # SET MODEL PARAMETERS
    batch_size = 1#100
    #learning_rate = 0.005
    evaluation_size = 500
    image_width = train_xdata[0].shape[0]
    image_height = train_xdata[0].shape[1]
    target_size = max(train_labels) + 1
    num_channels = 1 # greyscale = 1 channel
    generations = 500
    eval_every = 1
    #conv1_features = 25
    #conv2_features = 50
    #max_pool_size1 = 2
    #max_pool_size2 = 2
    #fully_connected_size1 = 100

    # MODEL PLACEHOLDER
    x_input_shape = (None, image_width, image_height, num_channels)
    x_input = tf.placeholder(tf.float32, shape=x_input_shape)
    y_target = tf.placeholder(tf.int32, shape=(None))

    eval_input_shape = (None, image_width, image_height, num_channels)
    eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
    eval_target = tf.placeholder(tf.float32, shape=(None))

    # CONVOLUTIONAL LAYER VARIABLES
    conv1_W = tf.Variable(tf.truncated_normal([conv1_filter, conv1_filter,
                                               num_channels, conv1_depth],
                                              stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
    conv1_b = tf.Variable(tf.zeros([conv1_depth], dtype=tf.float32))

    conv2_W = tf.Variable(tf.truncated_normal([conv2_filter, conv2_filter,
                                               conv1_depth, conv2_depth],
                                              stddev=0.1, dtype=tf.float32)) # 4x4 filter shape
    conv2_b = tf.Variable(tf.zeros([conv2_depth], dtype=tf.float32))

    fully1_W = tf.Variable(tf.truncated_normal([240, fully_size1], stddev=0.1, dtype=tf.float32))
    fully1_b = tf.Variable(tf.truncated_normal([fully_size1], stddev=0.1, dtype=tf.float32))

    fully2_W = tf.Variable(tf.truncated_normal([fully_size1, target_size], stddev=0.1, dtype=tf.float32))
    fully2_b = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

    model_output = font_CNN_model(x_input,
                                  conv1_W, conv1_b, max_pool_size1,
                                  conv2_W, conv2_b, max_pool_size2,
                                  fully1_W, fully1_b,
                                  fully2_W, fully2_b)
    #test_model_output = font_CNN_model(eval_input)

    # LOSS FUNCTION (softmax cross entropy)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
    
    # PREDICTION FUNCTION
    prediction = tf.nn.softmax(model_output)
    #test_prediction = tf.nn.softmax(test_model_output)

    # OPTIMIZER
    my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = my_optimizer.minimize(loss)
    
    ## INITIALIZE
    init = tf.global_variables_initializer()
    sess.run(init)

    # TRAIN START
    train_loss = []
    train_acc = []
    test_acc = []

    for i in range(generations):
        for sample_x, sample_y in tqdm(zip(train_xdata, train_labels)):
            font_train_data = np.expand_dims(sample_x, axis=0)
            font_train_data = np.expand_dims(font_train_data, axis=-1)
            train_dict = {x_input: font_train_data, y_target: [sample_y]}
            sess.run(train_step, feed_dict=train_dict)
            #temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)

        if (i+1) % eval_every == 0:
            eval_index = np.random.choice(len(train_xdata), size=batch_size)
            test_data = np.expand_dims(train_xdata[eval_index[0]], 0)
            test_data = np.expand_dims(test_data, -1)
            test_label = train_labels[eval_index[0]]
            temp_train_loss, temp_train_preds = sess.run(
                [loss, prediction],
                feed_dict={x_input: test_data, y_target: [test_label]})
            temp_train_acc = get_accuracy(temp_train_preds, test_label)

            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc]
            #print('Generation # {}. Train Loss: {}. Train Acc: {}'.format(*acc_and_loss))
            print(test_label, np.argmax(temp_train_preds))

    #print(np.random.choice(len(train_xdata), size=batch_size))
'''
train_xdata = np.expand_dims(train_xdata, axis=3)
    #a = sess.run(model_output, feed_dict={x_input: train_xdata})
    a = sess.run(loss, feed_dict={x_input: train_xdata, y_target: train_labels})
    #print(np.shape(a))
    print(a)
'''
