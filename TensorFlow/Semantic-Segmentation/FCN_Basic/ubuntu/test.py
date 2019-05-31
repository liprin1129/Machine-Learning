"""
Create batches of training data
:param batch_size: Batch Size
:return: Batches of training data
"""
from glob import glob
import os
import re
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

data_folder = '/mnt/14BC1C68BC1C4720/Development/Personal_Dev/Machine-Learning/TensorFlow/Semantic-Segmentation/CarND-Semantic-Segmentation/data/data_road/training'

batch_size = 10
image_shape = (160, 576)

image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
label_paths = {
    re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
    for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

background_color = np.array([255, 0, 0])

random.shuffle(image_paths)

for batch_i in range(0, len(image_paths), batch_size):
    images = []
    gt_images = []
    for image_file in image_paths[batch_i:batch_i+batch_size]:
        gt_image_file = label_paths[os.path.basename(image_file)]
        
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        
        print(image.shape, gt_image.shape)
        
        '''
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        
        ax[0].imshow(image)
        ax[1].imshow(np.invert(gt_image))
        plt.show()
        '''
        
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        #print(gt_bg.shape)
        
        gt_bg_2 = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        fig, ax = plt.subplots(3, 1, figsize=(10, 7))
        
        ax[0].imshow(image)
        ax[1].imshow(gt_bg[:, :, 0], cmap='gray')
        ax[2].imshow(np.invert(gt_bg)[:, :, 0], cmap='gray')
        plt.show()
        
        '''
        images.append(image)
        gt_images.append(gt_image)
        '''
        break
    break
