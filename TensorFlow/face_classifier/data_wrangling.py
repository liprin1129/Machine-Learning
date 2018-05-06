import cv2
import numpy as np
import os # for listing files in a directory
import re
from tqdm import tqdm
import pickle

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

        input_size = os.path.getsize(path+name)
        print("input_size: ", input_size)
        print("\t=> Load '{0}' to '{1}'".format(name, path))
        
        with open(path+name, "rb") as f:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f.read(max_bytes)

        data = pickle.loads(bytes_in)
        
        return data

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
        
        #jpg_list_with_path = []
        #jpg_list_without_path = []
        features = []
        labels = []
        
        for path, dnames, fnames in tqdm(os.walk(dir_path)):
            for img_file in fnames:
                if rx.search(img_file):
                    #jpg_list_with_path.extend([os.path.join(path, img_file)]) #jpg_list_with_path.extend([os.path.join(path, x) for x in fnames if rx.search(x)])
                    img_cv = self.read_img_with_abs_path(os.path.join(path, img_file)) 
                    if img_cv is not None:
                        features.append(img_cv)
                        
                    #jpg_list_without_path.append(img_file[:-9])
                    labels.append(img_file[:-9])

        #print(np.shape(features), np.shape(labels))
        return features, labels
        
    def read_img_with_abs_path(self, abs_file_name, channel = 0):
        ## read image
        return cv2.imread(abs_file_name, cv2.IMREAD_UNCHANGED)
        '''
        if channel == 0:
            self._img = cv2.imread(self._folder_path + file_name, cv2.IMREAD_GRAYSCALE)
        else:
            self._img = cv2.imread(self._folder_path + file_name, cv2.IMREAD_COLOR)
        #self._folder_path = folder_path
        '''
        
if __name__=="__main__":
    data_dir =  "/Users/pure/Developments/Personal-Study/Machine-Learning/Data/Face-Data/"
    '''
    data_wrangling = DataWrangling(data_dir)
    features, labels = data_wrangling.img2numpy_array(data_wrangling.dir_path)

    PickleHelper.save_to_pickle(data_dir, "faces-features.pkl", features)
    PickleHelper.save_to_pickle(data_dir, "faces-labels.pkl", labels)
    '''
    features = PickleHelper.load_pickle(data_dir, "faces-features.pkl")
    print(np.shape(features))
