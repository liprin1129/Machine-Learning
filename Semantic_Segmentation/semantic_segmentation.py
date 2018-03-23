import os.path
import tensorflow as tf
#import helper
import warnings
from distutils.version import LooseVersion
#import project_tests as tests

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

def show_tensor_name(sess, vgg_path):
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    op = graph.get_operations()
    for m in op:
        print(m.values())

sess = tf.Session()
show_tensor_name(sess, './vgg/')
