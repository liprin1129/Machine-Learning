# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:17:23 2017

@author: 170
"""

import cv2
import numpy as np

class ImgFunctions(object):
    def __init__(self, folder_path):
        self._dir_path = folder_path
        ## this must be set to inherit this class
        self._img = None

    @classmethod
    def read_img_with_abs_path(cls, abs_file_name):
        ## read image
        return cv2.imread(abs_file_name, cv2.IMREAD_UNCHANGED)

    @classmethod
    def bgr2rgb(cls, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #return image[...,::-1]

    @classmethod
    def bgr2gray(cls, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @classmethod
    def resize_img(cls, img, size=(32, 32)):
        return cv2.resize(img, dsize=size, interpolation = cv2.INTER_AREA)

    @classmethod
    def scailing(cls, img, new_min = 0, new_max = 1):
        new_img = img.copy()
        new_img = cv2.normalize(img, dst=None, alpha=new_min, beta=new_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return new_img


