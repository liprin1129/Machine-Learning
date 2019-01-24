import tensorflow as tf
import tensorflow_hub as hub
import os 

from Dataset import Dataset
n_class = 27

tf.reset_default_graph()

params = {
    'BATCH_SIZE': 64,
    'TOP_K': 5,                                 # How many top classes should be predicted

    'INFER_PATH': './Data/ImgsResize',
    'LABEL_PATH': './Data/classes.txt'
}

dataset = Dataset(params)
print("======> dataset.image_data: ", dataset.img_data)
#module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
#logits = module(dict(images=dataset.img_data))
#print(logits)

#softmax = tf.nn.softmax(logits)
#top_predictions = tf.nn.top_k(softmax, top_k, name='top_predictions')

module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1',
                    trainable=True)   # Trainable is True since we are going to fine-tune the model
print("\n========> output info dict:")
[print('{0}: {1}'.format(k, v)) for k, v in sorted(module.get_output_info_dict(signature='image_classification').items())]

module_features = module(dict(images=dataset.img_data), signature="image_classification",
                         as_dict=True)
#features = module_features["default"]
features = module_features["resnet_v1_50/block2"]

print("\n==========> features: ", features)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # making sure Tensorflow doesn't overflow the GPU

session = tf.InteractiveSession(config=config)

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','third')):
    os.mkdir(os.path.join('summaries','third'))

summ_writer = tf.summary.FileWriter(os.path.join('summaries','third'), session.graph)
'''
with tf.variable_scope('CustomLayer'):
    weight = tf.get_variable('weights', initializer=tf.truncated_normal((2048, n_class)))
    bias = tf.get_variable('bias', initializer=tf.ones((n_class)))
    logits = tf.nn.xw_plus_b(features, weight, bias)

# Find out the names of all variables present in graph
print(tf.all_variables())

# After finding the names of variables or scope, gather the variables you wish to fine-tune
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')
var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module/resnet_v2_50/block4')
var_list.extend(var_list2)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # making sure Tensorflow doesn't overflow the GPU

session = tf.InteractiveSession(config=config)

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','second')):
    os.mkdir(os.path.join('summaries','second'))

summ_writer = tf.summary.FileWriter(os.path.join('summaries','second'), session.graph)
'''