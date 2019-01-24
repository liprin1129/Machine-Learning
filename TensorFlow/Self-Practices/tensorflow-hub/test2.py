import tensorflow as tf
import tensorflow_hub as hub
import os

tf.reset_default_graph()
images = tf.placeholder(tf.float32, (None, 224, 224, 3))

"""
module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
logits = module(dict(images=images))
"""

"""
module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1',
                    trainable=True)   # Trainable is True since we are going to fine-tune the model
module_features = module(dict(images=images), signature="image_classification",
                         as_dict=True)
#features = module_features["default"]
logits = module_features["resnet_v1_50/block2"]

softmax = tf.nn.softmax(logits)
top_predictions = tf.nn.top_k(softmax, 2, name='top_predictions')

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','5')):
    os.mkdir(os.path.join('summaries','5'))

with tf.Session() as session:
    summ_writer = tf.summary.FileWriter(os.path.join('summaries','5'), session.graph)
"""

#######################################

"""
n_class = 2

module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
                    trainable=True)   # Trainable is True since we are going to fine-tune the model
module_features = module(dict(images=images), signature="image_feature_vector",
                         as_dict=True)
features = module_features["default"]

with tf.variable_scope('CustomLayer'):
    weight = tf.get_variable('weights', initializer=tf.truncated_normal((2048, n_class)))
    bias = tf.get_variable('bias', initializer=tf.ones((n_class)))
    logits = tf.nn.xw_plus_b(features, weight, bias)

# Find out the names of all variables present in graph
print(tf.all_variables())

# After finding the names of variables or scope, gather the variables you wish to fine-tune
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')
var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module/resnet_v2_50/block3')
var_list.extend(var_list2)

# Pass this set of variables into your optimiser
#optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=var_list)

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','5')):
    os.mkdir(os.path.join('summaries','5'))

with tf.Session() as session:
    summ_writer = tf.summary.FileWriter(os.path.join('summaries','5'), session.graph)
"""