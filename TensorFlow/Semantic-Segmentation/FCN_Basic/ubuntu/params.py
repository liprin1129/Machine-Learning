'''
Created on Oct 16, 2018

@author: pure
'''

import tensorflow as tf
#from tensorflow.contrib import layers
from tqdm import tqdm
import scipy.misc
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import random



with tf.variable_scope("VGG16"):
    
    num_channels = 3
    num_classes = 22 # 22 classes in total, including index 0 is background and 21 is border
    
    learning_rate = 0.0009
    scale_factor = 1e-3;
    #input_ph = tf.placeholder(tf.float64, [None, None, None, num_channels])
    #label_ph = tf.placeholder(tf.float64, [None, None, None, num_classes])
    
    image_shape = (160, 160)
    epochs = 30
    batch_size = 5

    input_ph = tf.placeholder(tf.float64, [None, 160, 160, num_channels])
    label_ph = tf.placeholder(tf.float64, [None, 160, 160, num_classes])
    
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
    '''
    for img_name in tqdm(person_train):
        img_name = re.sub(r'\s', '', img_name)
        image = scipy.misc.imread(os.path.join(root_dir_path, 'JPEGImages/') + img_name +'.jpg')
        gt_image = scipy.misc.imread(os.path.join(root_dir_path, 'SegmentationClass/') + img_name + '.png')
              
        scipy.misc.imsave('JPEG_Image/'+img_name+'.jpg', image)
        scipy.misc.imsave('Ground_Truth_Image/'+img_name+'.png', gt_image)
    '''

    '''
    def convert_mask_for_training(img_name):
        
        background_color = np.array([192, 128, 128])
        
        image = scipy.misc.imread(os.path.join(root_dir_path, 'JPEGImages/') + img_name +'.jpg')
        gt_image = scipy.misc.imread(os.path.join(root_dir_path, 'SegmentationClass/') + img_name + '.png')
        
        gt_bg = np.all(gt_image == background_color, axis = 2)
        
        gt_bg_reshape = gt_bg.reshape(*gt_bg.shape, 1)
        gt_bg_reshape_inv = np.invert(gt_bg_reshape)
        
        #if np.sum(np.all(gt_bg_reshape == gt_bg_reshape_inv, axis=2)) != 0 :
        gt_bg_concatenate = np.concatenate((gt_bg_reshape, gt_bg_reshape_inv), axis=2)
        
        #print('Ground true: ', np.sum(np.all(gt_bg_reshape == gt_bg_reshape_inv, axis=2)))
        print('[Image] shape: {0}, max: {1}, min: {2}'.format(np.shape(image), np.max(image), np.min(image)))
        print('[Ground true] shape: {0}, values: {1}'.format(np.shape(gt_bg_concatenate), set(gt_bg.reshape(-1))))
        
        _, ax = plt.subplots(3, 1, figsize=(10, 7))
        ax[0].imshow(image)
        ax[1].imshow(gt_bg, cmap='gray')
        ax[2].imshow(np.invert(gt_bg), cmap='gray')
        
        plt.show()
            
        return image, gt_bg_concatenate, gt_bg
        '''

    #jpeg_list = glob(os.path.join(root_dir_path, 'JPEGImages', '*.jpg'))
    gt_list = glob(os.path.join(root_dir_path, 'SegmentationClass', '*.png'))

    def batch_seperator(batch_size):
        for batch_idx in range(0, len(gt_list), batch_size):
            first = True

            for gt in gt_list[batch_idx:batch_idx+batch_size]:
                gt_png = Image.open(gt)
                #print(np.shape(gt_png))
                gt_name = os.path.basename(gt)

                img_name = re.sub(r'png', 'jpg', gt_name)
                img_jpeg = Image.open(os.path.join(root_dir_path, 'JPEGImages', img_name))
                #print(np.shape(img_jpeg))

                # Resize images
                img_jpeg = img_jpeg.resize(image_shape, Image.ANTIALIAS)
                gt_png = gt_png.resize(image_shape, Image.ANTIALIAS)
                gt_palette = gt_png.getpalette()
                
                # Convert to ndarray
                img_jpeg = np.array(img_jpeg, dtype=np.uint8)/255.0
                gt_png = np.array(gt_png, dtype=np.uint8)
                
                # Convert border line to class 21
                gt_png = np.where(gt_png==255, 21, gt_png)
                #print(set(gt_png.reshape(-1)))
                
                # Index 0 is background and 21 is border, 22 classes in total
                gt_hot = np.eye(num_classes)[gt_png]

                if first == True:
                    in_imgs = img_jpeg[np.newaxis, ...]
                    gt_imgs = gt_hot[np.newaxis, ...]
                    first = False
                else:
                    #print(in_imgs.shape, img_jpeg[np.newaxis].shape)
                    in_imgs = np.vstack((in_imgs, img_jpeg[np.newaxis, ...]))
                    gt_imgs = np.vstack((gt_imgs, gt_hot[np.newaxis, ...]))

            '''
            print(in_imgs.shape, gt_imgs.shape)

            for i in range(num_classes):
                _, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(in_imgs[0])
                ax[1].imshow(gt_imgs[0, :, :, i])
                plt.show()

            break
            '''
            yield in_imgs, gt_imgs