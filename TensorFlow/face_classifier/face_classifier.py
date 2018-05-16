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

    img_cv = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-features.pkl")
    img_label = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-labels.pkl")
    print("\nFEATURE SHAPE: {0}, LABEL SHAPE: {1}\n".format(np.shape(img_cv), np.shape(img_label)))
	
    '''
    # JUST FOR TEST
    np.random.seed(32)
    img_label = np.random.randint(2, size=len(img_label))
    print(img_label)
    #fc.imshow(img_cv[0])
	'''
	
    #fc.imshow(img_cv[0])
	
    vgg16 = VGG16(img_cv, img_label)
    #vgg16.architecture()
    vgg16.run_architecture()
