'''
Created on Oct 16, 2018

@author: pure
'''

import tensorflow as tf
#from tensorflow.contrib import layers
#from tqdm import tqdm
import scipy.misc
import os
import numpy as np
#import re
#import matplotlib.pyplot as plt

with tf.variable_scope("VGG16"):
    
    num_channels = 3
    num_classes = 2
    
    learning_rate = 0.0009
    input_ph = tf.placeholder(tf.float64, [None, None, None, num_channels])
    label_ph = tf.placeholder(tf.float64, [None, None, None, num_classes])
    
    #output_shape = tf.zeros(label_ph.get_shape().as_list(), tf.float64)
    #input_ph = tf.placeholder(tf.float64, [None, 334, 500, num_channels])
    #label_ph = tf.placeholder(tf.float64, [None, 334, 500, num_classes])
    
    '''
    conv_weights = {
        'cw1': tf.get_variable('CW1', shape=(3,3,3,64), initializer=layers.xavier_initializer()),
        'cw2': tf.get_variable('CW2', shape=(3,3,64,64), initializer=layers.xavier_initializer()),
        'cw3': tf.get_variable('CW3', shape=(3,3,64,128), initializer=layers.xavier_initializer()),
        'cw4': tf.get_variable('CW4', shape=(3,3,128,128), initializer=layers.xavier_initializer()),
        'cw5': tf.get_variable('CW5', shape=(3,3,128,256), initializer=layers.xavier_initializer()),
        'cw6': tf.get_variable('CW6', shape=(3,3,256,256), initializer=layers.xavier_initializer()),
        'cw7': tf.get_variable('CW7', shape=(1,1,256,256), initializer=layers.xavier_initializer()),
        'cw8': tf.get_variable('CW8', shape=(3,3,256,512), initializer=layers.xavier_initializer()),
        'cw9': tf.get_variable('CW9', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw10': tf.get_variable('CW10', shape=(1,1,512,512), initializer=layers.xavier_initializer()),
        'cw11': tf.get_variable('CW11', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw12': tf.get_variable('CW12', shape=(3,3,512,512), initializer=layers.xavier_initializer()),
        'cw13': tf.get_variable('CW13', shape=(1,1,512,512), initializer=layers.xavier_initializer())
        }
    
    conv_biases = {
        'cb1': tf.get_variable('CB1', shape=(64), initializer=layers.xavier_initializer()),
        'cb2': tf.get_variable('CB2', shape=(64), initializer=layers.xavier_initializer()),
        'cb3': tf.get_variable('CB3', shape=(128), initializer=layers.xavier_initializer()),
        'cb4': tf.get_variable('CB4', shape=(128), initializer=layers.xavier_initializer()),
        'cb5': tf.get_variable('CB5', shape=(256), initializer=layers.xavier_initializer()),
        'cb6': tf.get_variable('CB6', shape=(256), initializer=layers.xavier_initializer()),
        'cb7': tf.get_variable('CB7', shape=(256), initializer=layers.xavier_initializer()),
        'cb8': tf.get_variable('CB8', shape=(512), initializer=layers.xavier_initializer()),
        'cb9': tf.get_variable('CB9', shape=(512), initializer=layers.xavier_initializer()),
        'cb10': tf.get_variable('CB10', shape=(512), initializer=layers.xavier_initializer()),
        'cb11': tf.get_variable('CB11', shape=(512), initializer=layers.xavier_initializer()),
        'cb12': tf.get_variable('CB12', shape=(512), initializer=layers.xavier_initializer()),
        'cb13': tf.get_variable('CB13', shape=(512), initializer=layers.xavier_initializer())
        }
    
    conv_trans_weights = {
        'ctw13': tf.get_variable('CTW13', shape=(4,4,512,512), initializer=layers.xavier_initializer()),
        #'ctw13_add': tf.get_variable('CTW13', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        #'ctw10': tf.get_variable('CTW10', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        'ctw10_add': tf.get_variable('CTW10', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        #'ctw7': tf.get_variable('CTW7', shape=(4,4,128,256), initializer=layers.xavier_initializer()),
        'ctw7_add': tf.get_variable('CTW7', shape=(4,4,128,256), initializer=layers.xavier_initializer()),
        #'ctw4': tf.get_variable('CTW4', shape=(4,4,64,128), initializer=layers.xavier_initializer()),
        'ctw4_add': tf.get_variable('CTW4', shape=(4,4,64,128), initializer=layers.xavier_initializer()),
        #'ctw2': tf.get_variable('CTW2', shape=(4,4,2,64), initializer=layers.xavier_initializer()),
        'ctw2_add': tf.get_variable('CTW2', shape=(4,4,1,64), initializer=layers.xavier_initializer())
        }
    '''
    
    kernel_size = {
        '1x1': [1, 1],
        '2x2': [2, 2], 
        '3x3': [3, 3],
        '4x4': [4, 4],
        '8x8': [8, 8]
        }
    
    kernel_depth = {
        '1': 1,
        '64': 64,
        '128': 128,
        '256': 256,
        '512': 512
        }
    
    conv_strides = {
        '1x1': [1, 1],
        '2x2': [2, 2],
        '8x8': [8, 8]
        }
    
    pooling_strides = {
        '1x1': [1, 1, 1, 1],
        '2x2': [1, 2, 2, 1],
        '8x8': [1, 8, 8, 1]
        }
    
    pool_layers = [2, 4, 7, 10, 13]
    
    pool_size = {
        '1x1': [1, 1, 1, 1],
        '2x2': [1, 2, 2, 1]
        }
    
    epoch = 100

    ## Mac
    # image_list_path = '/Users/pure/Developments/Personal-Study/Machine-Learning/Data/VOC2012/'
    
    ## Ubuntu
    root_dir_path = '/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/Data/Segmentation/VOC2012/'
    
    ''' # Write image names to txt, which are about person image and exist in train.txt file. 
    with open(os.path.join(root_dir_path, 'ImageSets/Segmentation/', 'train.txt'), 'r') as rt:
        train_list = rt.readlines()
        
    with open(os.path.join(root_dir_path, 'ImageSets/Main/', 'person_train.txt'), 'r') as rt:
        train_person_list = rt.readlines()
    
    #print(len(train_list), len(train_person_list))
    
    
    jpg_re = []
    for name in tqdm(train_list):
        jpg_re.append(re.sub(r'\s', '', name))
    
    
    with open('/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/Data/Segmentation/VOC2012/person.txt', 'a') as fw:
        for person in train_person_list:
            person_re = re.sub(r'\s+(\-[0-9]|[0-9])\s', '', person)
            
            if (int(person[-3:-1]) >= 1) and (person_re in jpg_re):
                #print(person_re)
                fw.write(person_re+'\n')
    '''

    # READ PERSON.TXT FILE
    with open(os.path.join(root_dir_path, 'person.txt'), 'r') as fr:
        person_train = fr.readlines()
            
    def convert_mask_for_training(img_name):
        
        background_color = np.array([192, 128, 128])
        
        image = scipy.misc.imread(os.path.join(root_dir_path, 'JPEGImages/') + img_name +'.jpg')
        gt_image = scipy.misc.imread(os.path.join(root_dir_path, 'SegmentationClass/') + img_name + '.png')
        
        gt_bg = np.all(gt_image == background_color, axis = 2)
        
        gt_bg_reshape = gt_bg.reshape(*gt_bg.shape, 1)
        gt_bg_reshape_inv = np.invert(gt_bg_reshape)
        
        #if np.sum(np.all(gt_bg_reshape == gt_bg_reshape_inv, axis=2)) != 0 :
        gt_bg_concatenate = np.concatenate((gt_bg_reshape, gt_bg_reshape_inv), axis=2)
        
        '''
        #print('Ground true: ', np.sum(np.all(gt_bg_reshape == gt_bg_reshape_inv, axis=2)))
        print('[Image] shape: {0}, max: {1}, min: {2}'.format(np.shape(image), np.max(image), np.min(image)))
        print('[Ground true] shape: {0}, values: {1}'.format(np.shape(gt_bg_concatenate), set(gt_bg.reshape(-1))))
        
        _, ax = plt.subplots(3, 1, figsize=(10, 7))
        ax[0].imshow(image)
        ax[1].imshow(gt_bg, cmap='gray')
        ax[2].imshow(np.invert(gt_bg), cmap='gray')
        
        plt.show()
        '''
            
        return image, gt_bg_concatenate, gt_bg