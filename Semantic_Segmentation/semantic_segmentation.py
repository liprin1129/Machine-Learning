import os.path
import tensorflow as tf
#import helper
import warnings
from datetime import timedelta
from distutils.version import LooseVersion
import project_tests as tests
from helper import gen_batch_function, save_inference_samples
import time

# CHECK TENSORFLOW VERSION
assert LooseVersion(tf.__version__) >= LooseVersion("1.0"), "Please use TensorFlow version 1.0 or newer.  You are using {}".format(tf.__version__)
print("TensorFlow Version: {}".format(tf.__version__))

# CHECK FOR A GPU
if not tf.test.gpu_device_name():
    warnings.warn("No GPU found. Please use a GPU to train your neural network.")
else:
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    vgg_tag = "vgg16"
    vgg_input_tensor_name = "image_input:0"
    vgg_keep_prob_tensor_name = "keep_prob:0"
    vgg_layer3_out_tensor_name = "layer3_out:0"
    vgg_layer4_out_tensor_name = "layer4_out:0"
    vgg_layer7_out_tensor_name = "layer7_out:0"

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

def show_tensor_name(sess, vgg_path):
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    op = graph.get_operations()
    for m in op:
        print(m.values())

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    print('vgg3',vgg_layer3_out.get_shape())
    print('vgg4',vgg_layer4_out.get_shape())
    print('vgg7',vgg_layer7_out.get_shape())

    layer7_conv_1x1 = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    output = tf.layers.conv2d_transpose(
        layer7_conv_1x1,
        num_classes,
        4,
        2,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    layer4_conv_1x1 = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        1,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    output = tf.add(output, layer4_conv_1x1)
    output = tf.layers.conv2d_transpose(
        output,
        num_classes,
        4,
        2,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    layer3_conv_1x1 = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        1,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    output = tf.add(output, layer3_conv_1x1)
    output = tf.layers.conv2d_transpose(
        output,
        num_classes,
        16,
        8,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    return output
print("Layers Test:")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
print("Optimize Test:")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, data_dir):
    for epoch in range(epochs):
        s_time = time.time()
        for image, targets in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = {input_image: image,
                                            correct_label:targets,
                                            keep_prob: 0.5,
                                            learning_rate: 1e-4})

        print("Epoch: {}".format(epoch+1), "/ {}".format(epochs),
              " Loss: {:.5f}".format(loss),
              " Time: ", str(timedelta(seconds=(time.time() - s_time))))

        if (epoch + 1)%5 == 0:
            save_path = saver.save(sess, os.path.join(data_dir, 'epoch_' + str(epoch+1) + \
                                                      '.ckpt'))


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    learning_rate = 1e-4
    
    #print("Test Dataset:")
    #tests.test_for_kitti_dataset(data_dir)

    #print(os.path.join(data_dir, 'vgg'))
    #print(os.path.join(data_dir, 'data_road/training'))

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype = tf.float32)
        
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TRAIN
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_nn(sess, 20, 16, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, saver, runs_dir)

        # SAVE
        save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, num_classes)

run()

#sess = tf.Session()
#show_tensor_name(sess, './vgg/')
#image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, './vgg/')
#layers(layer3_out, layer4_out, layer7_out, 3)
