'''
Created on Oct 16, 2018

@author: pure
'''

import tensorflow as tf
from tensorflow.contrib import layers

with tf.variable_scope("VGG16"):
    
    in_img_ph = tf.placeholder(tf.float32, [1, None, None, 3])
    label_ph = tf.placeholder(tf.float32, [1, None, None, 1])
    
    #in_img_ph = tf.placeholder("float", [None, 224, 224, 3])
    #label_ph = tf.placeholder("float", [None, 224, 224, 2])
    
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
        'add1': tf.get_variable('ADD1', shape=(4,4,512,512), initializer=layers.xavier_initializer()),
        #'ctw13_add': tf.get_variable('CTW13', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        #'ctw10': tf.get_variable('CTW10', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        'add2': tf.get_variable('ADD2', shape=(4,4,256,512), initializer=layers.xavier_initializer()),
        #'ctw7': tf.get_variable('CTW7', shape=(4,4,128,256), initializer=layers.xavier_initializer()),
        'add3': tf.get_variable('ADD3', shape=(4,4,128,256), initializer=layers.xavier_initializer()),
        #'ctw4': tf.get_variable('CTW4', shape=(4,4,64,128), initializer=layers.xavier_initializer()),
        'add4': tf.get_variable('ADD4', shape=(4,4,64,128), initializer=layers.xavier_initializer()),
        #'ctw2': tf.get_variable('CTW2', shape=(4,4,2,64), initializer=layers.xavier_initializer()),
        'add5': tf.get_variable('ADD5', shape=(4,4,3,64), initializer=layers.xavier_initializer()),
        'output': tf.get_variable('output', shape=(1,1,1,64), initializer=layers.xavier_initializer())
        }
    
    strides = {
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
    # image_list_path = '/Users/pure/Developments/Personal-Study/Machine-Learning/Data/VOC2012/ImageSets/Main/'
    # image_dir_path = '/Users/pure/Developments/Personal-Study/Machine-Learning/Data/VOC2012/JPEGImages/'
    # mask_dir_path = '/Users/pure/Developments/Personal-Study/Machine-Learning/Data/VOC2012/SegmentationClass/'
    
    ## Ubuntu
    image_list_path = '/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/Data/Segmentation/VOC2012/ImageSets/Main/'
    image_dir_path = '/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/Data/Segmentation/VOC2012/JPEGImages/'
    mask_dir_path = '/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/Data/Segmentation/VOC2012/SegmentationClass/'
    
    with open(image_list_path+'person_train.txt', 'r') as rt:
        train_person_list = rt.readlines()
        