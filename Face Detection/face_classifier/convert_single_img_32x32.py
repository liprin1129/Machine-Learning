import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_wrangling import DataWrangling
from data_wrangling import PickleHelper

if __name__=="__main__":
    #img_path = "../../Data/Face/.png"
    img_path = "non_test.png"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    dw = DataWrangling(img_path)
    img = dw.bgr2rgb(img)
    img = dw.resize_img(img)
    img = dw.scailing(img, 0, 1)

    img = np.expand_dims(img, axis=0)
    #new_img = np.vstack([img, img])
    new_img = img.copy()
    for i in range(500-1):
        new_img = np.vstack([new_img, img])
        
    print("SHAPE: ", np.shape(new_img), "MIN: {0}, MAX: {1}".format(img.min(), img.max()))

    save_dir = "../../Data/Face/"
    PickleHelper.save_to_pickle(save_dir, "non_test-32x32.pkl", new_img)
