import cv2
import numpy as np
from pickle_helper import PickleHelper
from image_helper import ImgFunctions
import matplotlib.pyplot as plt

img = ImgFunctions.read_img_with_abs_path("../../Data/Face/star_wars.jpg")
#img = ImgFunctions.read_img_with_abs_path("../../Data/Face/mission_impossible.jpg")
#img = ImgFunctions.read_img_with_abs_path("../../Data/Face/Gang.jpg")
#img = ImgFunctions.read_img_with_abs_path("../../Data/Face/Ermin-Gang.jpg")
#img = ImgFunctions.read_img_with_abs_path("../../Data/Face/Tsunematsu-Itamochi.jpg")

def cut_down_to_windows(img, kernel_size=[360, 360], strides=[30, 30]):
    '''
    Convert img size to HD size (1920x1080), and return a list of small patches of the enlarged image.
    '''

    return_list = []
    
    hd_img = ImgFunctions.resize_img(img, (1920, 1080))
    hd_img = ImgFunctions.bgr2rgb(hd_img)
    tx = 0
    ty = 0


    iter_x = int((1920-kernel_size[0])/strides[0])+1
    iter_y = int((1080-kernel_size[1])/strides[1])+1
    
    for i in range(iter_x):
        for j in range(iter_y):
            #print("{0}, {1}: {2}, {3}".format(ty, ty+kernel_size[1], tx, tx+kernel_size[0]))
                
            #img2 = cv2.rectangle(hd_img,(tx,ty),(tx+kernel_size[0],ty+kernel_size[1]),(0,255,0),5)
            #plt.imshow(ImgFunctions.bgr2rgb(img2))
            #plt.show()
            
            patch_img = ImgFunctions.resize_img(hd_img[ty:ty+kernel_size[1], tx:tx+kernel_size[0]])
            patch_img = ImgFunctions.scailing(patch_img)
            return_list.append(patch_img)

            ty += strides[0]
            
        tx += strides[1]
        ty = 0
    

    print("Return shape:", np.shape(return_list))
    #plt.imshow(ImgFunctions.bgr2rgb(return_list[-1]))
    #plt.show()
    return np.array(return_list)

PickleHelper.save_to_pickle("../../Data/Face/", "star_wars_360x360.pkl", cut_down_to_windows(img, (250, 250), (10, 10)))
#PickleHelper.save_to_pickle("../../Data/Face/", "mission_impossible_360x360.pkl", cut_down_to_windows(img, (250, 250), (10, 10)))
#PickleHelper.save_to_pickle("../../Data/Face/", "Gang_360x360.pkl", cut_down_to_windows(img, (250, 250), (10, 10)))
#PickleHelper.save_to_pickle("../../Data/Face/", "Ermin-Gang_360x360.pkl", cut_down_to_windows(img, (250, 250), (10, 10)))
#PickleHelper.save_to_pickle("../../Data/Face/", "Tsunematsu-Itamochi_360x360.pkl", cut_down_to_windows(img, (250, 250), (10, 10)))

#https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
#https://www.coursera.org/learn/convolutional-neural-networks/lecture/fF3O0/yolo-algorithm
