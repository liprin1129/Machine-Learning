'''
Created on May 25, 2018

@author: user170
'''
import cv2
import numpy as np
import os # for listing files in a directory
import re
from tqdm import tqdm
import pickle

import resource # for memory usage check

# For one hot encoding
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from Helpers.image_helper import ImgFunction
from Helpers.pickle_helper import PickleHelper

class DataWrangling(object):
    """Data wrangling process.
    
    Attributes:
        dir_path(): Return directory path
        img2numpy_array(dir_path): 
    """
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._path = []; self._dnames = []; self._fnames = [];
         
        # Member instances for face and cifar images
        self._face_feature_arr = None; self._face_label_arr = None
        self._cifar_feature_arr = None; self._cifar_label_arr = None
        
        # Member instances for merged image array
        self._merged_feature_arr = None; self._merged_label_arr = None
        
        self._merged_pseudo_label_arr = None; self._merged_pseudo_label_one_hot_arr = None
    def __del__(self):
        print('Destructor called: Data-wragling tasks has done.')
        
    @property
    def dir_path(self): return self._dir_path
    @property
    def paths(self): return self._path
    @property
    def dnames(self): return self._dnames
    @property 
    def fnames(self): return self._fnames
    @property
    def merged_featrue_arr(self): return self._merged_feature_arr
    @property
    def merged_label_arr(self): return self._merged_label_arr
    @property
    def merged_pseudo_label_arr(self): return self._merged_pseudo_label_arr
    @property
    def merged_pseudo_label_one_hot_arr(self): return self._merged_pseudo_label_one_hot_arr
    
    # ################################# #
    #               Methods             #
    # ################################# #
    
    def tree_search(self, dir_path):
        """ Search directory and file names recursively
        """
        
        # Search directory and file names recursively
        for path, dnames, fnames in os.walk(dir_path):
            self._path.append(path)
            self._dnames.append(dnames)
            self._fnames.append(fnames)
    
    def query_memory_usage(self):
        memory_in_killo_byte = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('Memory usage: {0} mb'.format(round(memory_in_killo_byte/1000, 2)))
        
    def stack_images_as_array(self, paths, img_names, img_ext="jpg"):
        """ Convert OpenCV image to numpy array in a root directory
        """
        
        rx = re.compile(r"\.(jpg)")
        
        features = []
        labels = []
        
        # Search directory and file names recursively
        for idx, [path, fnames] in enumerate(zip(paths, img_names)):
            #print(idx, path, fnames)
            for img_file in fnames:
                if rx.search(img_file):
                    #print(os.path.join(path, img_file))
                    img_cv = ImgFunction.read_img_with_abs_path(os.path.join(path, img_file)) 
                    if img_cv is not None:
                        #img_cv = self.bgr2gray(img_cv) # convert color to gray scale
                        img_cv = ImgFunction.resize_img(img_cv, (32, 32)) # resize the image
                        img_cv = ImgFunction.bgr2rgb(img_cv)
                        features.append(img_cv)

                    labels.append(img_file[:-9])
                    
            if idx%5000 == 0:
                print("({0} %) Image list shape: {1}, Label list shape: {2}"\
                      .format(round(idx/len(paths) * 100, 2), np.shape(features), np.shape(labels)))
                #self.query_memory_usage()
                
        print("({0} %) Image list shape: {1}, Label list shape: {2}"\
              .format(100, np.shape(features), np.shape(labels)))
        #self.query_memory_usage()
        
        
        self._face_feature_arr = np.array(features); self._face_label_arr = np.array(labels)
        
    def one_hot_encoding(self, labels):
        label_binarizer = LabelBinarizer()
        labels_one_hot = label_binarizer.fit_transform(labels)

        encoder = OneHotEncoder()
        encoder.fit(labels_one_hot)
        self._merged_pseudo_label_one_hot_arr = encoder.transform(labels_one_hot).toarray()

        #return label_one_hot

    def integrate_face_and_cifar(self, cifar_dir, cifar_train_name):
        '''Merge face image array and cifar array
        '''
         
        # Load pickle data
        #feature1 = PickleHelper.load_pickle(path = "../../Data/Face/", name = "faces-32x32-features.pkl")
        self._cifar_feature_arr = PickleHelper.load_pickle(path = cifar_dir, name = cifar_train_name)[b'data']
        self._cifar_label_arr = PickleHelper.load_pickle(path = cifar_dir, name = cifar_train_name)[b'coarse_labels']
        
        # Change channel of image to the last dimension
        self._cifar_feature_arr = np.reshape(self._cifar_feature_arr, (-1, 3, 32, 32))
        self._cifar_feature_arr = np.moveaxis(self._cifar_feature_arr, 1, 3)
        
        self._merged_feature_arr = np.vstack([self._face_feature_arr, self._cifar_feature_arr])
        self._merged_label_arr = np.hstack([self._face_label_arr, self._cifar_label_arr])
        
        print("Merged feature shape: {0}, Merged label shape: {1}"\
              .format(self._merged_feature_arr.shape, self._merged_label_arr.shape))
        #self.query_memory_usage()
        
    def pseudo_face_obj_classification_label(self, labels):
        pseudo_label = []
        
        for label in labels:
            if str.isdigit(label):
                pseudo_label.append(0)
            else:
                pseudo_label.append(1)
                
        self._merged_pseudo_label_arr = pseudo_label
    
    def shuffle_feature_and_label_array(self, features, labels):
        # # shuffle the dataset
        shuffle_idx = np.arange(len(features))
        np.random.shuffle(shuffle_idx)
        
        self._merged_feature_arr = features[shuffle_idx]
        self._merged_pseudo_label_one_hot_arr = labels[shuffle_idx]
        
    def wrangling1(self):
        '''   
        '''
    
        # Data wragline for concatenating face and cifar dataset
        # # Normalization for face images

        new_feature = []
        label_landmark = [] # for the center of face. [class (face:0, none:1), x, y]
        for f in tqdm(feature1):
            #f_gray = self.bgr2gray(f)
            f_norm = ImgFunction.scailing(f, new_min=0, new_max=1)
            #print("feature1 < min: {0} | max: {1} >".format(np.min(f_norm), np.max(f_norm)))
            new_feature.append(f_norm)
            label_landmark.append([0, 16., 16.])
        feature1 = new_feature

        # # Change channel of image to the last dimension
        feature2 = np.reshape(feature2, (-1, 3, 32, 32))
        feature2 = np.moveaxis(feature2, 1, 3)
        
        # # Normalization for object images
        new_feature = []
        for f in tqdm(feature2):
            #f_gray = self.bgr2gray(f)
            f_norm = ImgFunction.scailing(f, new_min=0, new_max=1)
            #print("feature1 < min: {0} | max: {1} >".format(np.min(f_norm), np.max(f_norm)))
            new_feature.append(ImgFunction.scailing(f_norm))
            label_landmark.append([1, None, None])
        
        feature2 = new_feature
        
        features = np.vstack([feature1, feature2])
        
        # For simple classification [1 (face), 2 (none)]
        label1 = np.ones(len(feature1), dtype=np.float32)
        label2 = np.ones(len(feature2), dtype=np.float32)*2
        labels = np.hstack([label1, label2])
        
        # # shuffle the dataset
        shuffle_idx = np.arange(len(features))
        np.random.shuffle(shuffle_idx)
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        label_landmark = np.array(label_landmark)
        label_landmark = label_landmark[shuffle_idx]
        
        print("Data shape: {0}, Landmark shape: {1}".format(features.shape, labels.shape))
        #print("Data shape: {0}, Landmark shape: {1}".format(features.shape, label_landmark.shape))
        labels = self.one_hot_encoding(labels)
        
        #print(labels[:10])
        
        PickleHelper.save_to_pickle(save_dir, "faces-obj-32x32-features-norm.pkl", features)
        PickleHelper.save_to_pickle(save_dir, "faces-obj-32x32-labels.pkl", labels)
        #PickleHelper.save_to_pickle(save_dir, "faces-obj-32x32-labels-landmark.pkl", label_landmark)

    def wrangling_augmentation(self):
        features = PickleHelper.load_pickle(path = "../../Data/Face/Face-Obj-Augment/", name = "faces-obj-32x32-features-norm.pkl")
        labels = PickleHelper.load_pickle(path = "../../Data/Face/Face-Obj-Augment/", name = "faces-obj-32x32-labels.pkl")
        #labels = PickleHelper.load_pickle(path = "../../Data/Face/", name = "faces-obj-32x32-labels-landmark.pkl")
        
        print("Data shape: {0}, Landmark shape: {1}".format(features.shape, labels.shape))

        margin = list(range(-12, 13))

        for idx in tqdm(range(len(features))):

            # IMAGE ZOOM AND ZOOM OUT
            if np.random.randint(2) == 1:
                scale = np.random.randint(16, 48)
                resized_img = ImgFunction.resize_img(features[idx],
                                                      size=(scale, scale))
                #print(resized_img.shape, " | ", scale)

                padding = int((32-scale)/2)
                if padding >= 0:
                    resized_img = cv2.copyMakeBorder(resized_img,
                                                     int((32-scale)/2), int((32-scale)/2),
                                                     int((32-scale)/2), int((32-scale)/2),
                                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

                    features[idx] = ImgFunction.resize_img(resized_img)

                else:
                    resized_img = resized_img[int((scale-32)/2):int((scale-32)/2)+32, int((scale-32)/2):int((scale-32)/2)+32]
                    features[idx] = resized_img

                    '''
                # IMAGE TRANSLATE BY MARGIN 8
                if labels[idx][0] == 0: # if label is for face
                    
                    margin_rnd_idx = np.random.randint(len(margin))
                    features[idx] = ImgFunction.translation(features[idx], margin[margin_rnd_idx], margin[margin_rnd_idx])
                    
                    labels[idx][1] = labels[idx][1]+margin[margin_rnd_idx]
                    labels[idx][2] = labels[idx][2]+margin[margin_rnd_idx]
                    '''
                    '''
                    import matplotlib.pyplot as plt
                    plt.imshow(features[idx])
                    print(labels[idx])
                    plt.show()
                    '''

        PickleHelper.save_to_pickle(save_dir, "augment_faces-obj-32x32-features-norm.pkl", features)
        PickleHelper.save_to_pickle(save_dir, "augment_faces-obj-32x32-labels-landmark.pkl", labels)


def merge_face_obj_data():

    read_dir = "/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/Face-data/"
    save_dir = "/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/Face-wrangling/1. Face-Obj-Merge"
    
    # Data wrangling for raw face dataset
    dw = DataWrangling(read_dir)
    dw.tree_search(read_dir)
    dw.stack_images_as_array(dw.paths, dw.fnames, "jpg")

    dw.integrate_face_and_cifar(
        cifar_dir="/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Objects/cifar-100-python", 
        cifar_train_name="train")
    
    dw.pseudo_face_obj_classification_label(dw._merged_label_arr)
    dw.one_hot_encoding(dw.merged_pseudo_label_arr)
    #print(dw.merged_pseudo_label_one_hot_arr.shape)
    
    dw.shuffle_feature_and_label_array(dw.merged_featrue_arr, dw._merged_pseudo_label_one_hot_arr)
    
    PickleHelper.save_to_pickle(save_dir, "face-obj-32x32-features.pkl", dw.merged_featrue_arr)
    PickleHelper.save_to_pickle(save_dir, "face-obj-32x32-labels.pkl", dw._merged_pseudo_label_one_hot_arr)


if __name__=="__main__":
    merge_face_obj_data()
    #dw.wrangling1()
    #dw.wrangling_augmentation()
    '''
    a = PickleHelper.load_pickle(path = "/mnt/SharedData/Development/Personal_Dev/Machine-Learning/Data/Face/Face-wrangling", 
                             name = "faces-obj-32x32-features-norm.pkl")
    print(a.shape)
    '''