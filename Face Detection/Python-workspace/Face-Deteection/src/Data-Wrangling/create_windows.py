'''
Created on May 25, 2018

@author: user170
'''

import cv2
import numpy as np
from Helpers.pickle_helper import PickleHelper
from Helpers.image_helper import ImgFunction
import matplotlib.pyplot as plt
from tqdm import tqdm


class CreatWindows(object):

    def __init__(self, img_name):
        self._img = ImgFunction.read_img_with_abs_path(img_name, gray=False)
        
    def seperate_img_to_windows(self, kernel_size=[360, 360], strides=[30, 30]):
        '''
        Convert img size to HD size (1920x1080), and return a list of small patches of the enlarged image.
        '''
    
        windows_list = []
        
        hd_img = ImgFunction.resize_img(self._img, (1920, 1080))
        hd_img = ImgFunction.bgr2rgb(hd_img)
        
        tx = 0
        ty = 0
    
        iter_x = int((1920-kernel_size[0])/strides[0])+1
        iter_y = int((1080-kernel_size[1])/strides[1])+1
        
        for i in tqdm(range(iter_x)):
            for j in range(iter_y):
                
                patch_img = ImgFunction.resize_img(hd_img[ty:ty+kernel_size[1], tx:tx+kernel_size[0]])
                patch_img = ImgFunction.scailing(patch_img)
                windows_list.append(patch_img)
    
                ty += strides[0]
                
            tx += strides[1]
            ty = 0
        
    
        print("Return shape:", np.shape(windows_list))
        return np.array(windows_list)

if __name__ == "__main__":
    img_name = "/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/star_wars.jpg"
    cw = CreatWindows(img_name)
    cw.seperate_img_to_windows()
    