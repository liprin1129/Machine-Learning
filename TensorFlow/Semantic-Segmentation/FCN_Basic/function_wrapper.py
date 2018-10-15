'''
Created on Oct 13, 2018

@author: pure
'''
import tensorflow as tf

def convolution_2d(in_data_4d_tensor, weight_4d_tensor, bias_int, conv_strides_4arr, padding_type_str, relu_bool, name_str):
    new_name = name_str + '/conv'
    conv = tf.nn.conv2d(in_data_4d_tensor, weight_4d_tensor, strides=conv_strides_4arr, padding=padding_type_str, name=new_name)
    
    new_name = new_name + '/bias'
    conv_add_bias = tf.nn.bias_add(conv, bias_int, name=new_name)
    
    if relu_bool == True:
        new_name = new_name + '/relu'
        return tf.nn.relu(conv_add_bias, name=new_name)
    else:
        return conv_add_bias