'''
Created on Oct 21, 2018

@author: pure
'''

import tensorflow as tf
import params
import cv2
import tqdm
import numpy as np

#from tensorflow.contrib import layers

#in_img_ph = tf.placeholder("float", [1, None, None, 3])
#label_ph = tf.placeholder("float", [1, None, None, 2])

in_img_ph = tf.placeholder("float", [None, 224, 224, 3])
label_ph = tf.placeholder("float", [None, 224, 224, 2])

# ************* #
#    Layers    #
# ************* #
with tf.variable_scope("VGG16"):
    for i in range(13):
        pre_idx = i
        current_idx = i+1
        
        # ******************** #
        # Convolutional Layers #
        # ******************** #
        with tf.variable_scope('conv_layer{0}'.format(current_idx)):
            if current_idx is 1:
                globals()['layer{0}'.format(current_idx)] = tf.nn.conv2d(in_img_ph, 
                                                                         filter=params.conv_weights["cw{0}".format(current_idx)], 
                                                                         strides=params.strides['1x1'], 
                                                                         padding='SAME')
            else:
                globals()['layer{0}'.format(current_idx)] = tf.nn.conv2d(globals()['layer{0}'.format(pre_idx)], 
                                                                         filter=params.conv_weights["cw{0}".format(current_idx)], 
                                                                         strides=params.strides['1x1'], 
                                                                         padding='SAME')
                
            globals()['layer{0}'.format(current_idx)] = tf.nn.bias_add(globals()['layer{0}'.format(current_idx)], 
                                                                       params.conv_biases["cb{0}".format(current_idx)])
            globals()['layer{0}'.format(current_idx)] = tf.nn.relu(globals()['layer{0}'.format(current_idx)])
            
            if current_idx in params.pool_layers:
                globals()['layer{0}'.format(current_idx)] = tf.nn.avg_pool(globals()['layer{0}'.format(current_idx)], 
                                                                           ksize=params.pool_size['2x2'], 
                                                                           strides=params.strides['2x2'], 
                                                                           padding='SAME')
                
            print(globals()['layer{0}'.format(current_idx)])
            
# ****************************** #
# Convolutional Transpose Layers #
# ****************************** #
    with tf.variable_scope('conv_transpose'):
        with tf.variable_scope('layer13'):
            layer_trans = tf.nn.conv2d_transpose(globals()['layer13'], 
                                                 filter=params.conv_trans_weights['ctw13'], 
                                                 output_shape=[-1, int(globals()['layer10'].shape[1]), int(globals()['layer10'].shape[2]), 512],
                                                 strides=params.strides['2x2'], padding='SAME') #output_shape = [-1, 14, 14, 512],
                                                 
            
            #layer13_trans = tf.layers.conv2d_transpose(globals()['layer13'], 2, 4,  
            #                             strides= (2, 2), 
            #                             padding= 'same', 
            #                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
            #                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            print(layer_trans)
        with tf.variable_scope('layer10'):    
            print('Shape layer10: ', globals()['layer10'].shape[1])
            #print('\n'.join(globals()))
            layer_trans = tf.add(layer_trans, globals()['layer10'])
            layer_trans = tf.nn.conv2d_transpose(layer_trans, 
                                                 filter=params.conv_trans_weights['ctw10_add'], 
                                                 output_shape=[-1, int(globals()['layer10'].shape[1]), int(globals()['layer10'].shape[2]), 256],
                                                  strides=params.strides['2x2'], padding='SAME')
            
            print(layer_trans)

        with tf.variable_scope('layer7'):
            #layer10_trans = tf.nn.conv2d_transpose(globals()['layer10'], filter=params.conv_trans_weights['ctw10'], output_shape=[-1, 112, 112, 256],
            #                                       strides=params.strides['2x2'], padding='SAME')
            
            #layer4_trans = tf.layers.conv2d_transpose(globals()['layer4'], 2, 4,  
            #                             strides= (2, 2), 
            #                             padding= 'same', 
            #                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
            #                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            layer_trans = tf.add(layer_trans, globals()['layer7'])
            layer_trans = tf.nn.conv2d_transpose(layer_trans, filter=params.conv_trans_weights['ctw7_add'], output_shape=[-1, 56, 56, 128],
                                                   strides=params.strides['2x2'], padding='SAME')
            
            print(layer_trans)
            
        with tf.variable_scope('layer4'):
            #layer_trans = tf.nn.conv2d_transpose(globals()['layer3'], filter=params.conv_trans_weights['ctw3'], output_shape=[-1, 224, 224, 128],
            #                                       strides=params.strides['2x2'], padding='SAME')
            
            #layer3_trans = tf.layers.conv2d_transpose(globals()['layer3'], 2, 4, strides=(2, 2), padding='same', 
            #                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            layer_trans= tf.add(layer_trans, globals()['layer4'])
            layer_trans = tf.nn.conv2d_transpose(layer_trans, filter=params.conv_trans_weights['ctw4_add'], output_shape=[-1, 112, 112, 64],
                                                   strides=params.strides['2x2'], padding='SAME')
            
            print(layer_trans)
            
        with tf.variable_scope('layer2'):
            #layer_trans = tf.nn.conv2d_transpose(globals()['layer3'], filter=params.conv_trans_weights['ctw3'], output_shape=[-1, 224, 224, 128],
            #                                       strides=params.strides['2x2'], padding='SAME')
            
            #layer3_trans = tf.layers.conv2d_transpose(globals()['layer3'], 2, 4, strides=(2, 2), padding='same', 
            #                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            layer_trans= tf.add(layer_trans, globals()['layer2'])
            layer_trans = tf.nn.conv2d_transpose(layer_trans, filter=params.conv_trans_weights['ctw2_add'], output_shape=[-1, 224, 224, 2],
                                                   strides=params.strides['2x2'], padding='SAME')
            
            print(layer_trans)
            
            # L2 Regularizer
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/l2_regularizer
            
# ************ #
# Optimization #
# ************ #
    with tf.variable_scope('optimization'):
        logits = tf.reshape(layer_trans, (-1, 2))
        label = tf.reshape(label_ph, (-1,2))
        #print(logits)
        #print(label)
        # define loss function
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels= label))
        # define training operation
        optimizer = tf.train.AdamOptimizer(learning_rate= 0.0009)
        train_op = optimizer.minimize(cross_entropy_loss)
        
        print(train_op)
        
# ******** #
# Training #
# ******** #        
    with tf.variable_scope('training'):
        for epoch in tqdm(range(params.epoch)):
            for img_name in params.train_person_list:
                if int(img_name[-3:-1]) >= 1:
                    img_with_path = params.image_dir_path + img_name[:11] + '.jpg'
                    mask_with_path = params.mask_dir_path + img_name[:11] + '.png'
                    
                    # read mask image
                    mask_cv = cv2.imread(mask_with_path)
                    if mask_cv is not None: 
                        # read jpg image
                        img_cv = cv2.imread(img_with_path)
                        
                        
                    
                    