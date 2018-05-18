import numpy as np
import tensorflow as tf

from tqdm import tqdm

import platform

if platform.system() is "Darwin":
        import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
else:
	import matplotlib.pyplot as plt

import cv2

from tensorflow.contrib.layers import flatten

class Face_Classifier_With_Tensorflow(object):
	def __init__(self):
                

def face_classifier_did_loaded():
        print("Hello, World!")

plt.show()
