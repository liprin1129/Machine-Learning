import cv2
import numpy as np
import os # for listing files in a directory
import re
from tqdm import tqdm
import pickle

# For one hot encoding
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from image_helper import ImgFunctions
from pickle_helper import PickleHelper
'''
class PickleHelper(object):
    @classmethod
    def validation_check(cls, _path, _name):
        assert (_path or _name is not None), "Error: set corret path and name."
                
        if not _path.endswith("/"):
            return _path + "/"
            
        else:
            return _path
    
    @classmethod
    def save_to_pickle(cls, path = None, name = None, data = None):
        n_bytes = 2**31
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(data)
        
        path = cls.validation_check(path, name)

        with open(path+name, "wb") as f:
            for idx in range(0, len(bytes_out), max_bytes):
                print("\t => Save '{0}' to '{1}'".format(name, path))
                f.write(bytes_out[idx:idx+max_bytes])
                #pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    @classmethod        
    def load_pickle(cls, path = None, name = None):
        max_bytes = 2**31 - 1
        bytes_in = bytearray(0)
        
        path = cls.validation_check(path, name)
        print(path)
        
        input_size = os.path.getsize(path+name)
        print("input_size: ", input_size)
        print("\t=> Load '{0}' to '{1}'".format(name, path))
        
        with open(path+name, "rb") as f:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f.read(max_bytes)

        data = pickle.loads(bytes_in, encoding='bytes')
        
        return data
'''

class DataWrangling(object):
    # "/Users/pure/Developments/Personal-Study/Machine-Learning/Data/Face-Data/"
    def __init__(self, dir_path):
        self.__dir_path = dir_path

    @property
    def dir_path(self):
        return self.__dir_path

    def img2numpy_array(self, dir_path):
    #def read_dir(self, dir_path):
        rx = re.compile(r"\.(jpg)")
        
        features = []
        labels = []
        
        for path, dnames, fnames in tqdm(os.walk(dir_path)):
            for img_file in tqdm(fnames):
                if rx.search(img_file):

                    img_cv = ImgFunctions.read_img_with_abs_path(os.path.join(path, img_file)) 
                    if img_cv is not None:
                        #img_cv = self.bgr2gray(img_cv) # convert color to gray scale
                        img_cv = ImgFunctions.resize_img(img_cv, (32, 32)) # resize the image
                        img_cv = ImgFunctions.bgr2rgb(img_cv)
                        features.append(img_cv)

                    labels.append(img_file[:-9])

        #print(np.shape(features), np.shape(labels))
        return np.array(features), np.array(labels)
        
    def one_hot_encoding(self, labels):
        label_binarizer = LabelBinarizer()
        labels_one_hot = label_binarizer.fit_transform(labels)

        encoder = OneHotEncoder()
        encoder.fit(labels_one_hot)
        label_one_hot = encoder.transform(labels_one_hot).toarray()

        return label_one_hot

    # load face and cifar-100 data, then merge it with landmark labels for face
    def wrangling1(self):
        ''' # # face image to pickle
        features, labels = self.img2numpy_array(self.dir_path)
        
        PickleHelper.save_to_pickle(save_dir, "faces-32x32-features.pkl", features)
        PickleHelper.save_to_pickle(save_dir, "faces-32x32-labels.pkl", labels)    
        '''
        
        # Load pickle data
        feature1 = PickleHelper.load_pickle(path = "../../Data/Face/", name = "faces-32x32-features.pkl")
        feature2 = PickleHelper.load_pickle(path = "../../Data/Objects/cifar-100-python/", name = "train")[b'data']
    
        # Data wragline for concatenating face and cifar dataset
        # # Normalization for face images

        new_feature = []
        label_landmark = [] # for the center of face. [class (face:0, none:1), x, y]
        for f in tqdm(feature1):
            #f_gray = self.bgr2gray(f)
            f_norm = ImgFunctions.scailing(f, new_min=0, new_max=1)
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
            f_norm = ImgFunctions.scailing(f, new_min=0, new_max=1)
            #print("feature1 < min: {0} | max: {1} >".format(np.min(f_norm), np.max(f_norm)))
            new_feature.append(ImgFunctions.scailing(f_norm))
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
                resized_img = ImgFunctions.resize_img(features[idx],
                                                      size=(scale, scale))
                #print(resized_img.shape, " | ", scale)

                padding = int((32-scale)/2)
                if padding >= 0:
                    resized_img = cv2.copyMakeBorder(resized_img,
                                                     int((32-scale)/2), int((32-scale)/2),
                                                     int((32-scale)/2), int((32-scale)/2),
                                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

                    features[idx] = ImgFunctions.resize_img(resized_img)

                else:
                    resized_img = resized_img[int((scale-32)/2):int((scale-32)/2)+32, int((scale-32)/2):int((scale-32)/2)+32]
                    features[idx] = resized_img

                    '''
                # IMAGE TRANSLATE BY MARGIN 8
                if labels[idx][0] == 0: # if label is for face
                    
                    margin_rnd_idx = np.random.randint(len(margin))
                    features[idx] = ImgFunctions.translation(features[idx], margin[margin_rnd_idx], margin[margin_rnd_idx])
                    
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


if __name__=="__main__":
    read_dir = "../../Data/Face/lfw-truncated/"
    save_dir = "../../Data/Face/Face-Obj-Augment/"
    
    # Data wrangling for raw face dataset
    data_wrangling = DataWrangling(read_dir)
    #data_wrangling.wrangling1()
    data_wrangling.wrangling_augmentation()
