import numpy as np
from data_wrangling import PickleHelper
import matplotlib.pyplot as plt
from image_helper import ImgFunctions
'''
test_data_p = PickleHelper.load_pickle("../../Data/Face/", "blob-itamochi-32x32.pkl")
test_data_n = PickleHelper.load_pickle("../../Data/Face/", "non_test-32x32.pkl")

import numpy as np

a = np.vstack([test_data_n, test_data_p])
print(a.shape)

PickleHelper.save_to_pickle("../../Data/Face/", "itamochi-non_test-32x32.pkl", a)
'''

# To check image translation
data = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-features-norm.pkl")

print(data.shape)

import cv2

#cv2.imshow('tt', a)
#cv2.waitKey()
plt.imshow(ImgFunctions.translation(data[3], 8, 8))
plt.show()
