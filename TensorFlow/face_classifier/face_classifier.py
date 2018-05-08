import platform
if platform.system() == "Darwin":
    print("Platform is", platform.system())
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

import numpy as np

import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from VGG16 import VGG16

from data_wrangling import PickleHelper

class Face_Classifier_With_Tensorflow(object):
    def __init__(self):
        self.__path = None

    def imshow(self, img):
        plt.imshow(img[...,::-1], cmap="gray")
        plt.show()

def face_classifier_did_loaded():
    
    fc = Face_Classifier_With_Tensorflow()

    img_cv = PickleHelper.load_pickle("../../Data/", "faces-features.pkl")

    print(np.shape(img_cv))
    #fc.imshow(img_cv[0])

    '''
    img = cv2.imread("../../Data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = cv2.resize(img, dsize=(32,32), interpolation=cv2.INTER_LINEAR_EXACT)
    #fc.imshow(img)
    fc.imshow(img2)
    '''

    vgg16 = VGG16()
    vgg16.architecture(img_cv)
