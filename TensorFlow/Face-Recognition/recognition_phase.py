import tensorflow as tf
import tensorflow_deeplay as dlay
import os

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
                
                #short_cut_tensor = tf.multiply(short_cut_tensor, 0.1)

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
                
                #short_cut_tensor = tf.multiply(short_cut_tensor, 0.1)

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
                #short_cut_tensor = tf.multiply(short_cut_tensor, 0.1)

            #conv = dlay.unit_conv(conv, _output_ch_list=[64, 64, 256], _training=train_phase)
            rep_conv = _repetitive_layers()

            bottle_neck = tf.nn.relu(tf.add(short_cut_tensor, rep_conv))

    return bottle_neck


def facenet(_input, train_validation_phase=False):
    
    #train_phase = tf.placeholder(tf.bool);

    with tf.variable_scope("conv1"):
        #kernel_dict = {"_output_ch_int": 64, "_kernel_w_int": 3, "_kernel_h_int": 3}

        #print("\n===> ", params.kenel_output_ch_int, params.kernel_w_int, params.kernel_h_int)
        #pooling_dict = {"_window_shape": [2, 2], "_pooling_type": "MAX", "_pooling_padding": "SAME", "_strides": [2, 2]}
        
        conv = dlay.conv_layer_2d(input_img, _output_ch_int=64, _kernel_w_int=3, _kernel_h_int=3,
                                        _stride_list=[1, 2, 2, 1],
                                        _pooling_window_shape=[2,2], _pooling_type = "MAX", 
                                        _pooling_padding = "SAME", _pooling_strides = [2,2], _batch_norm=True, _train_val_phase=train_validation_phase)
        #print("\n===> ", conv)

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

    with tf.variable_scope("logits"):
        logits = tf.nn.pool(unit, 
                            window_shape= [unit.get_shape()[1], unit.get_shape()[2]], 
                            pooling_type="AVG",
                            padding="VALID",
                            strides=[1,1])
        logits = dlay.conv_layer_2d(logits, _output_ch_int=1001, _kernel_w_int=1, _kernel_h_int=1, _stride_list=[1, 1, 1, 1],
                                        _bias=True, _train_val_phase=train_validation_phase)
        logits = tf.squeeze(logits)
    return logits

if __name__=="__main__":
    input_img = tf.placeholder(tf.float32, (None, 224, 224, 3))

    conv = facenet(input_img, train_validation_phase=True)

    """
    print(conv)

    print("\n====> Global variable: ")
    [print(i) for i in tf.global_variables()]

    print("\n====> Local variable: ")
    [print(i) for i in tf.local_variables()]

    print("\n====> Trainable variable: ")
    [print(i) for i in tf.trainable_variables()]
    """
    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','facenet')):
        os.mkdir(os.path.join('summaries','face'))

    with tf.Session() as session:
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), session.graph)