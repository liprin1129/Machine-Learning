import tensorflow as tf
import os
#kernel_dict = {"_output_ch_int": None, "_kernel_w_int": None, "_kernel_h_int": None}
#pooling_dict = {"_window_shape": None, "_pooling_type": None, "_pooling_padding": None, "_strides": None}

def conv_layer_2d(_input_tensor, _kernel_dict, _stride_list, _conv_padding='SAME', _bias=None, _activation_fn=None, _pooling_dict=None):
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

    weight_var = tf.get_variable("weights", 
                            [_kernel_dict["_kernel_w_int"], _kernel_dict["_kernel_h_int"], 
                                _input_tensor.get_shape()[3], _kernel_dict["_output_ch_int"]], 
                            dtype=tf.float32, 
                            initializer=tf.random_normal_initializer(), 
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), 
                            trainable=True)

    # Convolutional node
    conv = tf.nn.conv2d(_input_tensor, weight_var, _stride_list, padding=_conv_padding)

    # Biase to be learned
    if _bias is not None:
        bias_var = tf.get_variable("bias", 
                                    _kernel_dict["_output_ch_int"], 
                                    dtype=tf.float32, 
                                    initializer=tf.random_normal_initializer(), 
                                    trainable=True)
        conv = tf.nn.bias_add(conv, bias_var)
    
    if _activation_fn is not None:
        conv = _activation_fn(conv)

    if _pooling_dict is not None:
        conv = tf.nn.pool(conv, 
                        window_shape=_pooling_dict["_window_shape"], 
                        pooling_type=_pooling_dict["_pooling_type"],
                        padding=_pooling_dict["_pooling_padding"],
                        strides=_pooling_dict["_strides"])
    return conv

def facenet(_input):

    with tf.variable_scope("entry_convolution"):
        kernel_dict = {"_output_ch_int": 64, "_kernel_w_int": 3, "_kernel_h_int": 3}
        pooling_dict = {"_window_shape": [2, 2], "_pooling_type": "MAX", "_pooling_padding": "SAME", "_strides": [2, 2]}

        conv = conv_layer_2d(input_img, kernel_dict, [1, 2, 2, 1], _bias=True, _pooling_dict=pooling_dict)

    with tf.variable_scope("block1"):
        #short_cut_tensor = tf.identity(conv)
        short_cut_tensor = conv

        with tf.variable_scope("unit1"):
            for i in range(3):
                with tf.variable_scope("conv{0}".format(i)):
                    kernel_dict = {"_output_ch_int": 64, "_kernel_w_int": 3, "_kernel_h_int": 3}
                    conv = conv_layer_2d(conv, kernel_dict, [1, 1, 1, 1])
                    print(conv)
            short_cut_tensor = tf.multiply(short_cut_tensor, 0.1)

            bottle_neck = tf.nn.relu(tf.add(short_cut_tensor, conv))

    return conv

if __name__=="__main__":
    input_img = tf.placeholder(tf.float32, (None, 128, 128, 3))

    conv = facenet(input_img)
    print(conv)

    print("\n====> Global variable: ")
    [print(i) for i in tf.global_variables()]

    print("\n====> Local variable: ")
    [print(i) for i in tf.local_variables()]

    print("\n====> Trainable variable: ")
    [print(i) for i in tf.trainable_variables()]

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','facenet')):
        os.mkdir(os.path.join('summaries','face'))

    with tf.Session() as session:
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), session.graph)