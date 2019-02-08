import sys; sys.path.append("/home/user170/shared-data/Personal_Dev/Machine-Learning/TensorFlow/Common_helper")

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
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _batch_norm_ph=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
                                                        _stride_list=[1, 1, 1, 1], _batch_norm_ph=_training)
                
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
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _batch_norm_ph=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = dlay.conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
                                                        _stride_list=[1, 1, 1, 1], _batch_norm_ph=_training)
                
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
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv2"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 2, 2, 1], _activation_fn=tf.nn.relu, _batch_norm_ph=_training)
        with tf.variable_scope("conv3"):
            rep_conv = dlay.conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _batch_norm_ph=_training)
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
                                        _pooling_padding = "SAME", _pooling_strides = [2,2], 
                                        _activation_fn=tf.nn.relu,
                                        _batch_norm_ph=train_validation_phase)
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
        logits = dlay.conv_layer_2d(logits, _output_ch_int=outputs_logits_int, _kernel_w_int=1, _kernel_h_int=1, _stride_list=[1, 1, 1, 1], _bias=True)
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

        with tf.variable_scope("placeholders/"):
            self.labels_dataset_placeholder = tf.placeholder(tf.int64, shape = [None], name="labels")
            self.handle_placeholder = tf.placeholder(tf.string, shape=[], name="iterator_handler")
            self.train_valid_placeholder = tf.placeholder(tf.bool, name="tv_mode_selector_placeholder")


        self.predictions = None
        self.optimizer = None
        self.accuracy = None
        self.optimizer = None
        self.loss = None
        self.loss_minimizer = None
        self.grads_and_vars = None

        self.train_iterator = None
        self.validation_iterator = None
        self.get_next_in_interators = None

    def __call__(self, _tfdata_dir, _height, _width):
        def _train_valid_iterator_setup():
            tfrecord_helper = TFRecord_Helper(_height, _width, verbose=False)
            with tf.variable_scope("placeholders/"): self.input_dataset_placeholder = tf.placeholder(tf.float32, shape = [None, _height, _width, 3], name="input")
            #self.input_dataset_placeholder = tf.placeholder(tf.float32, shape = [None, _height, _width, 3], "input_placeholder")

            with tf.variable_scope("iterator_handler"):
                with tf.variable_scope("train_iterator"):
                    self.train_iterator = tfrecord_helper.convert_from_tfrecord_with_tf_dataset(_tfdata_dir, self.batch, "train")
                
                with tf.variable_scope("validation_iterator"):
                    self.validation_iterator = tfrecord_helper.convert_from_tfrecord_with_tf_dataset(_tfdata_dir, self.batch, "valid")

                with tf.variable_scope("iterator"): 
                    iterator = tf.data.Iterator.from_string_handle(
                        self.handle_placeholder, (tf.float32, tf.int64), ([None, _height, _width, 3], [None]))

            self.get_next_in_interators = iterator.get_next()

        def _facenet_architecture_setup():
            logits = facenet(self.input_dataset_placeholder, self.train_valid_placeholder, self.output, self.block1, self.block2, self.block3, self.block4)

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_dataset_placeholder, logits = logits))

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.variable_scope("optimizer"), tf.control_dependencies(extra_update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                self.loss_minimizer = self.optimizer.minimize(self.loss)

            with tf.variable_scope("predictions"):
                self.predictions = tf.argmax(logits, 1, output_type = tf.int32)

            with tf.variable_scope("variance"):
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            
            #with tf.variable_scope("accuracy"):                
                #self.accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(self.predictions, tf.cast(self.labels_dataset_placeholder, tf.int32)), tf.float32)), self.batch) * 100
                #self.accuracy = tf.equal(tf.cast(self.predictions, tf.int32), tf.cast(self.labels_dataset_placeholder, tf.int32))

                #self.accuracy = np.sum(np.equal(self.predictions, self.labels_dataset_placeholder))*100.0#/self.labels_dataset_placeholder.get_shape().as_list()[0]
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!", self.predictions.get_shape())
                #self.accuracy = tf.equal(tf.shape(self.predictions), tf.shape(self.labels_dataset_placeholder))

        _train_valid_iterator_setup()
        _facenet_architecture_setup()