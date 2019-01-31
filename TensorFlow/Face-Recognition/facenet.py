import sys; sys.path.append("/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/Common_helper")

import tensorflow as tf
import tensorflow_deeplay as dlay
from tfrecorder_helper import TFRecord_Helper

import os
from tqdm import tqdm
import numpy as np

def unit_conv_shortcut(_input_tensor, unit_num_int, _output_ch_list=[], _training=None):
    assert _training is not None

    def _repetitive_layers():
        with tf.variable_scope("conv1"):
            rep_conv = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
                                                        _stride_list=[1, 1, 1, 1], _batch_norm=True, _train_val_phase=_training)
                
                #short_cut_tensor = tf.multiply(short_cut_tensor, 10)

            #conv = dlay.unit_conv(conv, _output_ch_list=[64, 64, 256], _training=train_phase)
            rep_conv = _repetitive_layers()

            bottle_neck = tf.nn.relu(tf.add(short_cut_tensor, rep_conv))

    return bottle_neck


def unit_conv_skip(_input_tensor, unit_num_int, _output_ch_list=[], _training=None):
    assert _training is not None

    def _repetitive_layers():
        with tf.variable_scope("conv1"):
            rep_conv = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
                                                        _stride_list=[1, 1, 1, 1], _batch_norm=True, _train_val_phase=_training)
                
                #short_cut_tensor = tf.multiply(short_cut_tensor, 10)

            #conv = dlay.unit_conv(conv, _output_ch_list=[64, 64, 256], _training=train_phase)
            rep_conv = _repetitive_layers()

            bottle_neck = tf.nn.relu(tf.add(short_cut_tensor, rep_conv))

    return bottle_neck


def unit_conv_maxpool(_input_tensor, unit_num_int, _output_ch_list=[], _training=None):
    assert _training is not None

    def _repetitive_layers():
        with tf.variable_scope("conv1"):
            rep_conv = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 2, 2, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = tf.nn.pool(_input_tensor, 
                                                window_shape=[2,2], 
                                                pooling_type="MAX",
                                                padding="VALID",
                                                strides=[2,2])
                #short_cut_tensor = tf.multiply(short_cut_tensor, 10)

            #conv = dlay.unit_conv(conv, _output_ch_list=[64, 64, 256], _training=train_phase)
            rep_conv = _repetitive_layers()

            bottle_neck = tf.nn.relu(tf.add(short_cut_tensor, rep_conv))

    return bottle_neck


def facenet(_input, train_validation_phase, outputs_logits_int, *layer_shape_dict):
    
    #train_validation_phase = tf.placeholder(tf.bool);

    with tf.variable_scope("conv1"):
        #kernel_dict = {"_output_ch_int": 64, "_kernel_w_int": 3, "_kernel_h_int": 3}

        #print("\n===> ", params.kenel_output_ch_int, params.kernel_w_int, params.kernel_h_int)
        #pooling_dict = {"_window_shape": [2, 2], "_pooling_type": "MAX", "_pooling_padding": "SAME", "_strides": [2, 2]}
        
        conv = dlay.conv_layer_2d(_input, _output_ch_int=64, _kernel_w_int=3, _kernel_h_int=3,
                                        _stride_list=[1, 2, 2, 1],
                                        _pooling_window_shape=[2,2], _pooling_type = "MAX", 
                                        _pooling_padding = "SAME", _pooling_strides = [2,2], _batch_norm=True, _train_val_phase=train_validation_phase)
        #print("\n===> ", conv)

    with tf.variable_scope("block1"):
        unit = unit_conv_shortcut(conv, unit_num_int=1, _output_ch_list=layer_shape_dict[0]["unit1"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=layer_shape_dict[0]["unit2"], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=3, _output_ch_list=layer_shape_dict[0]["unit3"], _training=train_validation_phase)

    with tf.variable_scope("block2"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=layer_shape_dict[1]["unit1"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=layer_shape_dict[1]["unit2"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=layer_shape_dict[1]["unit3"], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=4, _output_ch_list=layer_shape_dict[1]["unit4"], _training=train_validation_phase)

    with tf.variable_scope("block3"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=layer_shape_dict[2]["unit1"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=layer_shape_dict[2]["unit2"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=layer_shape_dict[2]["unit3"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=4, _output_ch_list=layer_shape_dict[2]["unit4"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=5, _output_ch_list=layer_shape_dict[2]["unit5"], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=6, _output_ch_list=layer_shape_dict[2]["unit6"], _training=train_validation_phase)

    with tf.variable_scope("block4"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=layer_shape_dict[3]["unit1"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=layer_shape_dict[3]["unit2"], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=layer_shape_dict[3]["unit3"], _training=train_validation_phase)
    """
    ###########################
    with tf.variable_scope("block1"):
        unit = unit_conv_shortcut(conv, unit_num_int=1, _output_ch_list=[64, 64, 256], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=[64, 64, 256], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=3, _output_ch_list=[64, 64, 256], _training=train_validation_phase)

    with tf.variable_scope("block2"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=[128, 128, 512], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=[128, 128, 512], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=[128, 128, 512], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=4, _output_ch_list=[128, 128, 512], _training=train_validation_phase)

    with tf.variable_scope("block3"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=4, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=5, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)
        unit = unit_conv_maxpool(unit, unit_num_int=6, _output_ch_list=[256, 256, 1024], _training=train_validation_phase)

    with tf.variable_scope("block4"):
        unit = unit_conv_shortcut(unit, unit_num_int=1, _output_ch_list=[512, 512, 2048], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=2, _output_ch_list=[512, 512, 2048], _training=train_validation_phase)
        unit = unit_conv_skip(unit, unit_num_int=3, _output_ch_list=[512, 512, 2048], _training=train_validation_phase)
    ###########################
    """
    with tf.variable_scope("logits"):
        logits = tf.nn.pool(unit, 
                            window_shape= [unit.get_shape()[1], unit.get_shape()[2]], 
                            pooling_type="AVG",
                            padding="VALID",
                            strides=[1,1])
        #logits = dlay.conv_layer_2d(logits, _output_ch_int=1001, _kernel_w_int=1, _kernel_h_int=1, _stride_list=[1, 1, 1, 1],
        logits = dlay.conv_layer_2d(logits, _output_ch_int=outputs_logits_int, _kernel_w_int=1, _kernel_h_int=1, _stride_list=[1, 1, 1, 1],
                                        _bias=True, _train_val_phase=train_validation_phase)
        logits = tf.squeeze(logits)
    return logits


class HyperParameters(object):
    def __init__(self, _num_epochs_int, _batch_size_int, _learning_rate_int, _num_outputs_int):
        self.block1 = {"unit1":[], "unit2":[], "unit3":[]}
        self.block2 = {"unit1":[], "unit2":[], "unit3":[], "unit4":[]}
        self.block3 = {"unit1":[], "unit2":[], "unit3":[], "unit4":[], "unit5":[], "unit6":[]}
        self.block4 = {"unit1":[], "unit2":[], "unit3":[]}

        self.epoch = _num_epochs_int
        self.batch = _batch_size_int
        self.learning_rate = _learning_rate_int
        self.output = _num_outputs_int

class FacenetModel(HyperParameters):
    def __init__(self, _num_epochs_int, _batch_size_int, _learning_rate_int, _num_outputs_int):
        HyperParameters.__init__(self, _num_epochs_int, _batch_size_int, _learning_rate_int, _num_outputs_int) # Parent class intializer

        self.train_valid_placeholder = tf.placeholder(tf.bool)

        self.predictions = None
        self.optimizer = None
        self.accuracy = None

        self.train_iterator = None
        self.validation_iterator = None        
        self.input_dataset_placeholder = None
        self.labels_dataset_placeholder = tf.placeholder(tf.int64, shape = [None])
        self.handle_placeholder = tf.placeholder(tf.string, shape=[])
        self.get_next_in_interators = None

    def __call__(self, _tfdata_dir, _height, _width):
        def _train_valid_iterator_setup():
            tfrecord_helper = TFRecord_Helper(_height, _width, verbose=False)

            self.input_dataset_placeholder = tf.placeholder(tf.float32, shape = [None, _height, _width, 3])

            with tf.variable_scope("iterator_handler"):
                with tf.variable_scope("train_iterator"):
                    self.train_iterator = tfrecord_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', self.batch, "train")
                
                with tf.variable_scope("validation_iterator"):
                    self.validation_iterator = tfrecord_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', self.batch, "valid")

                with tf.variable_scope("iterator"): 
                    iterator = tf.data.Iterator.from_string_handle(
                        self.handle_placeholder, (tf.float32, tf.int64), ([None, _height, _width, 3], [None]))

            self.get_next_in_interators = iterator.get_next()

        def _facenet_architecture_setup():
            logits = facenet(self.input_dataset_placeholder, self.train_valid_placeholder, self.output, self.block1, self.block2, self.block3, self.block4)

            with tf.variable_scope("loss"):
                loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_dataset_placeholder, logits = logits))

            with tf.variable_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)

            with tf.variable_scope("predictions"):
                self.predictions = tf.argmax(logits, 1, output_type = tf.int32)

            """
            with tf.variable_scope("accuracy"):                
                self.accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(self.predictions, tf.cast(self.labels_dataset_placeholder, tf.int32)), tf.float32)), self.batch) * 100
                #self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, tf.cast(self.labels_dataset_placeholder, tf.int32)), tf.float32))
                #self.accuracy = tf.equal(tf.cast(self.predictions, tf.int64), self.labels_dataset_placeholder)
                #self.accuracy = tf.equal(self.predictions, self.labels_dataset_placeholder)
                #print(self.predictions); sys.stdout.flush()
                #print(self.labels_dataset_placeholder); sys.stdout.flush()
            """
        _train_valid_iterator_setup()
        _facenet_architecture_setup()


if __name__=="__main__":
    epoch = 5
    batch = 20
    learning_rate = 0.001
    num_outputs = 9

    model = FacenetModel(epoch, batch, learning_rate, num_outputs)

    model.block1["unit1"] = [64, 64, 256]
    model.block1["unit2"] = [64, 64, 256]
    model.block1["unit3"] = [64, 64, 256]
    
    model.block2["unit1"] = [128, 128, 512]
    model.block2["unit2"] = [128, 128, 512]
    model.block2["unit3"] = [128, 128, 512]
    model.block2["unit4"] = [128, 128, 512]

    model.block3["unit1"] = [256, 256, 1024]
    model.block3["unit2"] = [256, 256, 1024]
    model.block3["unit3"] = [256, 256, 1024]
    model.block3["unit4"] = [256, 256, 1024]
    model.block3["unit5"] = [256, 256, 1024]
    model.block3["unit6"] = [256, 256, 1024]

    model.block4["unit1"] = [512, 512, 2048]
    model.block4["unit2"] = [512, 512, 2048]
    model.block4["unit3"] = [512, 512, 2048]

    model('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 224, 224)

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','facenet')):
        os.mkdir(os.path.join('summaries','face'))

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config_proto) as sess:
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), sess.graph)
        sess.run(tf.global_variables_initializer())

        training_handle = sess.run(model.train_iterator.string_handle())
        validation_handle = sess.run(model.validation_iterator.string_handle())
        for _ in tqdm(range(epoch)):
            try:
                print("=== Training ==="); sys.stdout.flush()
                sess.run(model.train_iterator.initializer)
                total_accuracy = 0
                count = 0
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: training_handle})
                    _ = sess.run(model.optimizer, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.labels_dataset_placeholder:extracted_data[1], model.train_valid_placeholder:True})

                    pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                    total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

                    #print("{0} accuracy: {1}".format(np.sum(equality), equality)); sys.stdout.flush()
                    #print("expected:\t{0} \npredicted:\t{1}".format(extracted_data[1], pred)); sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()
                pass

            try:
                print("=== Validation ==="); sys.stdout.flush()
                sess.run(model.validation_iterator.initializer)
                total_accuracy = 0
                count = 0
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: validation_handle})
                    pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                    total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

                    #equality = np.equal(pred, extracted_data[1])
                    #print("{0} accuracy: {1}".format(np.sum(equality), equality)); sys.stdout.flush()
                    #print("expected:\t{0} \npredicted:\t{1}".format(extracted_data[1], pred)); sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()
                pass


    """
    print(conv)

    print("\n====> Global variable: ")
    [print(i) for i in tf.global_variables()]

    print("\n====> Local variable: ")
    [print(i) for i in tf.local_variables()]

    print("\n====> Trainable variable: ")
    [print(i) for i in tf.trainable_variables()]
    """

