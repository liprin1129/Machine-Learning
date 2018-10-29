'''
Created on Oct 21, 2018

@author: pure
'''

import tensorflow as tf
import params
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#from tensorflow.contrib import layers

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
                globals()['layer{0}'.format(current_idx)] = tf.nn.conv2d(params.in_img_ph, 
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
                
            
            #print(globals()['layer{0}'.format(current_idx)])
    
    for i in range(13):
        current_idx = i+1
        print("Layer {0}: {1}".format(current_idx, globals()['layer{0}'.format(current_idx)]))
# ****************************** #
# Convolutional Transpose Layers #
# ****************************** #
    with tf.variable_scope('conv_transpose'):
        with tf.variable_scope('add_skip_1'):
            layer_trans = tf.nn.conv2d_transpose(globals()['layer13'], 
                                                 filter=params.conv_trans_weights['add1'], 
                                                 #output_shape=[-1, int(globals()['layer10'].shape[1]), int(globals()['layer10'].shape[2]), 512],
                                                 output_shape=[int(globals()['layer10'].shape[0]), -1, -1, 512],
                                                 strides=params.strides['2x2'], padding='SAME') #output_shape = [-1, 14, 14, 512],
                                                 
            
            #layer13_trans = tf.layers.conv2d_transpose(globals()['layer13'], 2, 4,  
            #                             strides= (2, 2), 
            #                             padding= 'same', 
            #                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
            #                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            print('Trans Conv: ', layer_trans.shape, globals()['layer10'].shape)
            
            layer_trans = tf.add(layer_trans, globals()['layer10'])
            
        
        with tf.variable_scope('add_skip_2'):    
            #print('Shape layer10: ', globals()['layer10'].shape[1])
            #print('\n'.join(globals()))
            layer_trans = tf.nn.conv2d_transpose(layer_trans,
                                                 filter=params.conv_trans_weights['add2'], 
                                                 #output_shape=[-1, int(globals()['layer7'].shape[1]), int(globals()['layer7'].shape[2]), 256],
                                                 output_shape=[int(globals()['layer7'].shape[0]), -1, -1, 256],
                                                 strides=params.strides['2x2'], padding='SAME')
            
            print('Trans Conv: ', layer_trans.shape, globals()['layer7'].shape)
            
            layer_trans = tf.add(layer_trans, globals()['layer7'])
            
        with tf.variable_scope('add_skip_3'):
            layer_trans = tf.nn.conv2d_transpose(layer_trans, 
                                                   filter=params.conv_trans_weights['add3'], 
                                                   #output_shape=[-1, int(globals()['layer4'].shape[1]), int(globals()['layer4'].shape[2]), 128],
                                                   output_shape=[int(globals()['layer4'].shape[0]), -1, -1, 128],
                                                   strides=params.strides['2x2'], padding='SAME')
            
            #layer4_trans = tf.layers.conv2d_transpose(globals()['layer4'], 2, 4,  
            #                             strides= (2, 2), 
            #                             padding= 'same', 
            #                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
            #                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            print('Trans Conv: ', layer_trans.shape, globals()['layer4'].shape)
            
            layer_trans = tf.add(layer_trans, globals()['layer4'])
        
        with tf.variable_scope('add_skip_4'):
            layer_trans = tf.nn.conv2d_transpose(layer_trans, 
                                                 filter=params.conv_trans_weights['add4'], 
                                                 #output_shape=[-1, int(globals()['layer2'].shape[1]), int(globals()['layer2'].shape[2]), 64],
                                                 output_shape=[int(globals()['layer2'].shape[0]), -1, -1, 64],
                                                 strides=params.strides['2x2'], padding='SAME')
            
            #layer3_trans = tf.layers.conv2d_transpose(globals()['layer3'], 2, 4, strides=(2, 2), padding='same', 
            #                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            print('Trans Conv: ', layer_trans.shape, globals()['layer2'].shape)
            
            layer_trans= tf.add(layer_trans, globals()['layer2'])

        with tf.variable_scope('output'):
            #layer_trans = tf.nn.conv2d_transpose(globals()['layer3'], filter=params.conv_trans_weights['ctw3'], output_shape=[-1, 224, 224, 128],
            #                                       strides=params.strides['2x2'], padding='SAME')
            
            #layer3_trans = tf.layers.conv2d_transpose(globals()['layer3'], 2, 4, strides=(2, 2), padding='same', 
            #                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
            
            layer_trans = tf.nn.conv2d_transpose(layer_trans, filter=params.conv_trans_weights['output'], output_shape=[-1, 224, 224, 1],
                                                   strides=params.strides['2x2'], padding='SAME')
            
            print('Trans Conv (output): ', layer_trans.shape)
            
            # L2 Regularizer
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/l2_regularizer
            
# ************ #
# Optimization #
# ************ #
    with tf.variable_scope('optimization'):
        logits = tf.reshape(layer_trans, (-1, 2))
        label = tf.reshape(params.label_ph, (-1,2))
        #print(logits)
        #print(label)
        # define loss function
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels= label))
        # define training operation
        optimizer = tf.train.AdamOptimizer(learning_rate= 0.0009)
        train_op = optimizer.minimize(cross_entropy_loss)
        
        #print(train_op)
        
# ******** #
# Training #
# ******** #        
    #with tf.variable_scope('training'):
    with tf.Session() as sess:
        for epoch in tqdm(range(100)):
            for img_name in params.train_person_list:
                if int(img_name[-3:-1]) >= 1:
                    img_with_path = params.image_dir_path + img_name[:11] + '.jpg'
                    mask_with_path = params.mask_dir_path + img_name[:11] + '.png'
                    
                    # read mask image
                    #mask_cv = cv2.cvtColor(cv2.imread(mask_with_path), cv2.COLOR_BGR2RGB)
                    mask_cv = cv2.imread(mask_with_path)
                    
                    if mask_cv is not None: 
                        # read jpg image
                        img_cv = cv2.imread(img_with_path)
                        
                        # Convert BGR to RBG
                        person_mask = np.zeros([mask_cv.shape[0], mask_cv.shape[1]])
                        
                        mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2RGB)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        #print(img_cv.shape, mask_cv.shape)
                        
                        # Change mask image pixel values to range [1, 0]
                        person_mask[mask_cv[:, :, 0] == 192] = 1.0 # R
                        '''
                        mask_cv[mask_cv[:, :, 1] != 128] = 0.0 # G
                        mask_cv[mask_cv[:, :, 2] != 128] = 0.0 # B
                        
                        mask_cv[mask_cv[:, :, 0] == 192] = 1.0 # R
                        mask_cv[mask_cv[:, :, 1] == 128] = 1.0 # G
                        mask_cv[mask_cv[:, :, 2] == 128] = 1.0 # B
                        '''
                        
                        ''' # See depth color set
                        for depth in range(3):
                            pixel = set()
                            for i in range(mask_cv.shape[0]):
                                for j in range(mask_cv.shape[1]):
                                    pixel.add(mask_cv[i, j, depth])
                            
                            print(depth, ': ', pixel)
                        '''
                        
                        ''' # Show images
                        plt.figure(1)
                        plt.subplot(211)
                        plt.imshow(img_cv)
                        
                        plt.subplot(212)
                        plt.imshow(mask_cv)
                        plt.show()
                        '''
                        #plt.imshow(person_mask*200, cmap='gray')
                        #plt.show()
                        
                        img_cv = img_cv[np.newaxis, :, :, :]
                        person_mask = person_mask[np.newaxis, :, :, np.newaxis]
                        print(img_cv.shape, ' || ', person_mask.shape)
                        
                        _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={params.in_img_ph: img_cv, params.label_ph: person_mask})
                        print("Loss: = {:.3f}".format(loss))
                        
                        #break