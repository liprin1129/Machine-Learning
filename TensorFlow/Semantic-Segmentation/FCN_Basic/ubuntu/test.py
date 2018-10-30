'''
Created on Oct 29, 2018

@author: pure
'''

import tensorflow as tf
import params
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


with tf.variable_scope("Convolutional"):
    print('*******************************************')
    print('*           Convolutional Layers          *')
    print('*******************************************')
    
    with tf.variable_scope("group1"):
        print()
        kernel_depth = '64'
        
        layer1 = tf.layers.conv2d(inputs=params.input_ph, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer1')
        print(layer1)
        
        layer2 = tf.layers.conv2d(inputs=layer1, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer2')
        print(layer2)
        
        layer3 = tf.nn.avg_pool(layer2, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer3')
        
        print(layer3)
        
    with tf.variable_scope("group2"):
        print()
        kernel_depth = '128'
        
        layer4 = tf.layers.conv2d(inputs=layer3, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer4')
        print(layer4)
        
        layer5 = tf.layers.conv2d(inputs=layer4, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer5')
        print(layer5)
        
        layer6 = tf.nn.avg_pool(layer5, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer6')
        print(layer6)
    
    with tf.variable_scope("group3"):
        print()
        kernel_depth = '256'
        
        layer7 = tf.layers.conv2d(inputs=layer6, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer7')
        print(layer7)
        
        layer8 = tf.layers.conv2d(inputs=layer7, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer8')
        print(layer8)
        
        layer9 = tf.layers.conv2d(inputs=layer8, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer9')
        print(layer9)
        
        layer10 = tf.nn.avg_pool(layer9, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer10')
        print(layer10)
        
    with tf.variable_scope("group4"):
        print()
        kernel_depth = '512'
        
        layer11 = tf.layers.conv2d(inputs=layer10, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer11')
        print(layer11)
        
        layer12 = tf.layers.conv2d(inputs=layer11, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer12')
        print(layer12)

        layer13 = tf.layers.conv2d(inputs=layer12, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),use_bias=True, name='layer13')
        print(layer13)
                
        layer14 = tf.nn.avg_pool(layer13, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer14')
        print(layer14)
        
    with tf.variable_scope("group5"):
        print()
        kernel_depth = '512'
        
        layer15 = tf.layers.conv2d(inputs=layer14, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer14')
        print(layer11)
        
        layer16 = tf.layers.conv2d(inputs=layer15, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer16')
        print(layer16)

        layer17 = tf.layers.conv2d(inputs=layer16, filters=params.kernel_depth[kernel_depth], 
                                  kernel_size=params.kernel_size['3x3'], strides=params.conv_strides['1x1'], 
                                  padding='same',activation=tf.nn.relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  use_bias=True, name='layer17')
        print(layer17)
                
        layer18 = tf.nn.avg_pool(layer17, ksize=params.pool_size['2x2'], strides=params.pooling_strides['2x2'], padding='SAME', name='layer18')
        print(layer18)

        
with tf.variable_scope('Transpose'):
    print('\n')
    print('*******************************************')
    print('*      Transpose Convolutional Layers     *')
    print('*******************************************')
    print()
    
    trans_weight1 = weights = tf.Variable(initial_value=tf.random_normal([4, 4, layer14.get_shape().as_list()[3], layer18.get_shape().as_list()[3]], 
                                                                         stddev=0.1, dtype=tf.float64), name="trans_weight1")
    l2_regularizer1 = tf.nn.l2_loss(trans_weight1)
    trans1 = tf.nn.conv2d_transpose(layer18, filter=trans_weight1, output_shape=tf.shape(layer14), strides=params.pooling_strides['2x2'], padding='SAME', name='trans1')
    trans1 = tf.nn.relu(trans1)
    print(trans1)

    trans_weight2 = weights = tf.Variable(initial_value=tf.random_normal([4, 4, layer10.get_shape().as_list()[3], layer14.get_shape().as_list()[3]], 
                                                                         stddev=0.1, dtype=tf.float64), name="trans_weight2")
    l2_regularizer2 = tf.nn.l2_loss(trans_weight2)
    trans2 = tf.nn.conv2d_transpose(layer14, filter=trans_weight2, output_shape=tf.shape(layer10), strides=params.pooling_strides['2x2'], padding='SAME', name='trans2')
    trans2 = tf.nn.relu(trans2)
    print(trans2)
    
    trans_weight3 = weights = tf.Variable(initial_value=tf.random_normal([4, 4, layer6.get_shape().as_list()[3], layer10.get_shape().as_list()[3]], 
                                                                         stddev=0.1, dtype=tf.float64), name="trans_weight3")
    l2_regularizer3 = tf.nn.l2_loss(trans_weight3)
    trans3 = tf.nn.conv2d_transpose(layer10, filter=trans_weight3, output_shape=tf.shape(layer6), strides=params.pooling_strides['2x2'], padding='SAME', name='trans3')
    trans3 = tf.nn.relu(trans3)
    print(trans3)
    
    trans_weight4 = weights = tf.Variable(initial_value=tf.random_normal([4, 4, layer3.get_shape().as_list()[3], layer6.get_shape().as_list()[3]], 
                                                                         stddev=0.1, dtype=tf.float64), name="trans_weight4")
    l2_regularizer4 = tf.nn.l2_loss(trans_weight4)
    trans4 = tf.nn.conv2d_transpose(layer6, filter=trans_weight4, output_shape=tf.shape(layer3), strides=params.pooling_strides['2x2'], padding='SAME', name='trans4')
    trans4 = tf.nn.relu(trans4)
    print(trans4)
    
    '''
    trans1 = tf.layers.conv2d_transpose(inputs=layer18, filters=layer14.get_shape().as_list()[3], kernel_size=(5, 5), 
                                        strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='trans1')
    print(trans1)
    
    trans2 = tf.layers.conv2d_transpose(inputs=layer14, filters=layer10.get_shape().as_list()[3], kernel_size=(4, 4), 
                                        strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='trans2')
    print(trans2)
    
    trans3 = tf.layers.conv2d_transpose(inputs=layer10, filters=layer6.get_shape().as_list()[3], kernel_size=(4, 4), 
                                        strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='trans3')
    print(trans3)
    
    trans4 = tf.layers.conv2d_transpose(inputs=layer6, filters=layer3.get_shape().as_list()[3], kernel_size=(4, 4), 
                                        strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='trans4')
    print(trans4)
    '''

with tf.variable_scope('skip_addition'):
    skip_addition1 = tf.add(trans1, layer14, name='skip_addition1')
    skip_addition2 = tf.add(trans2, layer10, name='skip_addition2')
    skip_addition3 = tf.add(trans3, layer6, name='skip_addition3')
    skip_addition4 = tf.add(trans4, layer3, name='skip_addition4')
    
    print('\n')
    print('*******************************************')
    print('*               Output Layer              *')
    print('*******************************************')
    print()
    
    trans_weight5 = weights = tf.Variable(initial_value=tf.random_normal([4, 4, params.num_classes, layer3.get_shape().as_list()[3]], 
                                                                         stddev=0.1, dtype=tf.float64), name="trans_weight4")
    l2_regularizer5 = tf.nn.l2_loss(trans_weight5)
    output_trans = tf.nn.conv2d_transpose(skip_addition4, filter=trans_weight5, output_shape=tf.shape(params.label_ph), strides=params.pooling_strides['2x2'], padding='SAME', name='output_trans')
    output_trans = tf.nn.relu(output_trans)
    print(output_trans)
    
    '''
    output_layer = tf.layers.conv2d_transpose(inputs=skip_addition4, filters=params.num_classes, kernel_size=(3, 3), 
                                        strides=params.conv_strides['2x2'], padding='same', activation=tf.nn.relu, use_bias=True, name='output')
    print(output_layer)
    '''


with tf.variable_scope('optimization'):
    print('\n')
    print('*******************************************')
    print('*               Optimization              *')
    print('*******************************************')
    print()
    #logits = tf.reshape(layer_trans, (-1, 2))
    #label = tf.reshape(params_backup.label_ph, (-1, 2))
    #print(logits)
    #print(label)
    # define loss function
    print("Learning Rate    : ", params.learning_rate)
    print("Optimizer        : ", 'Adam Optimizer')
    print("Softmax          : ", True)
    print("Cross Entropy    : ", True)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_trans, labels = params.label_ph)) 
    cross_entropy_loss = cross_entropy_loss + l2_regularizer1 + l2_regularizer2 + l2_regularizer3 + l2_regularizer4 + l2_regularizer5
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= params.learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    

#with tf.variable_scope('training'):
with tf.Session() as sess:
    print('\n')
    print('*******************************************')
    print('*                 Training                *')
    print('*******************************************')
    print()
    sess.run(tf.global_variables_initializer())
    
    for epoch in tqdm(range(100)):
        
        count = 0
        loss_avg = 0.0
        
        for img_name in tqdm(params.person_train):

            img_with_path = params.image_dir_path + img_name[:11] + '.jpg'
            mask_with_path = params.mask_dir_path + img_name[:11] + '.png'

            # read mask image
            img_cv = cv2.imread(img_with_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_cv = img_cv/255
        
            mask_cv = cv2.imread(mask_with_path)
            mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2RGB)
            
            # Convert BGR to RBG
            person_mask = np.ones([img_cv.shape[0], img_cv.shape[1]])
            #print(img_cv.shape, mask_cv.shape)
            
            # Change mask image pixel values to range [1, 0]
            person_mask[mask_cv[:, :, 0] != 192] = 0.0 # R
            person_mask[mask_cv[:, :, 1] != 128] = 0.0 # R
            person_mask[mask_cv[:, :, 2] != 128] = 0.0 # R

            '''
            # Show images
            plt.figure(1)
            plt.subplot(211)
            plt.imshow(img_cv)
            
            plt.subplot(212)
            plt.imshow(person_mask)
            plt.show()
            '''
            
            img_cv = img_cv[np.newaxis, :, :, :]
            person_mask = person_mask[np.newaxis, :, :, np.newaxis]
            #print(img_cv.shape, ' || ', person_mask.shape)
            
            #out = tf.nn.softmax(output_trans)
            
            pred, _, loss = sess.run([output_trans, train_op, cross_entropy_loss], feed_dict={params.input_ph: img_cv, params.label_ph: person_mask})
            #print("Loss: = {:.3f}".format(loss))
            
            #pred = sess.run(output_trans, feed_dict={params.input_ph: img_cv})
            #print("pred: ", pred[0, :, :, 0])
            #plt.imshow(pred)
            #plt.show()
            

            loss_avg += loss
            loss_avg /= (count+1)
            
            if (count+1)%20 == 0:
                print("\nTemporal Loss: = {0:.10f}".format(loss))
                print("Loss Mean: = {0:.10f}".format(loss_avg))
 
            count += 1
            #break
