import tensorflow as tf
    
def conv_layer_2d(_input_tensor, 
                    _output_ch_int, _kernel_w_int, _kernel_h_int,
                     _stride_list, _conv_padding='SAME', 
                    _bias=None, _activation_fn=None, 
                    _pooling_window_shape=None, _pooling_type = None, 
                    _pooling_padding = None, _pooling_strides = None,
                    _batch_norm = None,
                    _train_val_phase = None):
    """
    Create compact convolutional layer by combining tf.nn.conv2d, bias, activation, and pooling functions.
    
    Args:
        _input_tensor: 4 dimensional input tensor with shape [batch, width, height, channels].
        _kernel_dict: dictionary to make kernel (also so called weight) variable.
            _output_ch_int: output channel depth. This is used for the shape argument of tf.get_variable().
            _kernel_w_int: width of kernel
            _kernel_h_int: height of kernel.
        _kernel_var: A tf.Variable instance for kernel (or so called filter).
        _stride_list: A list of ints. 1-D tensor of length 4.
        _padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        _bias: True or False for bias addition.
        _activation_fn: Activation functions; sigmoid, tanh, elu, selu, softplus, softsign and etc.
        _pooling_dict: A Dictionary for tf.nn.pool() arguments.
            _window_shape: List of window shape.
            _pooling_type: "AVG" or "MAX"
            _padding: "SAME" or "VALID"
            _strides: List of stride.
    """
    #conv_param = ConvParams()
    #conv_param.output_ch_int = _output_ch_int
    #conv_param.kernel_w_int = _kernel_w_int
    #conv_param.kernel_h_int = _kernel_h_int
    #train_phase = tf.placeholder(tf.bool);

    weight_var = tf.get_variable("weights", 
                            [_kernel_w_int, _kernel_h_int, 
                                _input_tensor.get_shape()[3], _output_ch_int], 
                            dtype=tf.float32, 
                            initializer=tf.random_normal_initializer(), 
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), 
                            trainable=True)

    # Convolutional node
    with tf.variable_scope("conv2d"):
        conv = tf.nn.conv2d(_input_tensor, weight_var, _stride_list, padding=_conv_padding)

    # Biase to be learned
    if _bias is not None:
        with tf.variable_scope("bias_add"):
            bias_var = tf.get_variable("bias", 
                                        _output_ch_int, 
                                        dtype=tf.float32, 
                                        initializer=tf.random_normal_initializer(), 
                                        trainable=True)
            conv = tf.nn.bias_add(conv, bias_var)

    if _batch_norm is not None:
        with tf.variable_scope("batch_norm"):
            conv = tf.layers.batch_normalization(conv, training=_train_val_phase)

    if _activation_fn is not None:
        with tf.variable_scope("activation"):
            conv = _activation_fn(conv)

    if (_pooling_window_shape and _pooling_type and _pooling_padding and _pooling_strides) is not None:
        with tf.variable_scope("pooling"):
            conv = tf.nn.pool(conv, 
                            window_shape=_pooling_window_shape, 
                            pooling_type=_pooling_type,
                            padding=_pooling_padding,
                            strides=_pooling_strides)
            """conv = tf.nn.pool(conv, 
                            window_shape=_pooling_dict["_window_shape"], 
                            pooling_type=_pooling_dict["_pooling_type"],
                            padding=_pooling_dict["_pooling_padding"],
                            strides=_pooling_dict["_strides"])"""

    return conv

"""
def unit_conv_shortcut(_input_tensor, unit_num_int, _output_ch_list=[], _training=None):
    assert _training is not None

    def _repetitive_layers():
        with tf.variable_scope("conv1"):
            rep_conv = conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
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
            rep_conv = conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        return rep_conv

    with tf.variable_scope("unit{0}".format(unit_num_int)):
        with tf.variable_scope("bottlenet_v1"):
            #short_cut_tensor = tf.identity(conv)
            with tf.variable_scope("short_cut"):
                short_cut_tensor = conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3,
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
            rep_conv = conv_layer_2d(_input_tensor, _output_ch_int=_output_ch_list[0], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 1, 1, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv2"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[1], _kernel_w_int=3, _kernel_h_int=3, 
                                            _stride_list=[1, 2, 2, 1], _activation_fn=tf.nn.relu, _batch_norm=True, _train_val_phase=_training)
        with tf.variable_scope("conv3"):
            rep_conv = conv_layer_2d(rep_conv, _output_ch_int=_output_ch_list[2], _kernel_w_int=3, _kernel_h_int=3, 
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
"""