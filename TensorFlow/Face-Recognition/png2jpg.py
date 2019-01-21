import cv2
import sys
import numpy as np
import os
from tqdm import tqdm

class ImageHelper(object):
    """Helper class that provides TensorFlow image coding utilities.
    Args: 
        height: Prefered height of an output image. 
                Default is 15 pixel.
        width:  Prefered width of an output image. 
                Default is 15 pixel.
    """
    def __init__(self, height=15, width=15, verbose=False):
        self._height = height
        self._width = width
        self._channels = 3
        self._image_format = 'jpg'
        #self._image_data = None
        self._verbose_img = verbose

    def cv_read_img_with_abs_path(self, abs_file_name):
        image = cv2.imread(abs_file_name, cv2.IMREAD_UNCHANGED)
        if self._verbose_img==True: print("1. Number of channels: ", image.shape); sys.stdout.flush()

        if image.shape[2] > self._channels: # if image is in RGBD format
            image = self.cv_bgra2rgb(image)
        elif image.shape[2] == self._channels:
            image = self.cv_bgr2rgb(image)
        
        #if image.shape[0]!=15 and image.shape[1]!=15: # default image size, 15X15
        #    image = cv2.resize(image, dsize=(self._width, self._height), interpolation = cv2.INTER_AREA)

        if self._verbose_img==True: print("2. Number of channels: ", image.shape); sys.stdout.flush()

        return image.astype(np.float32)

    def cv_bgra2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    def cv_bgr2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2RGB)

    def _get_filenames_and_classes(self, dataset_dir):
        """Returns a list of filenames and inferred class names.
        Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.
        Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
        """
        image_root = os.path.join(dataset_dir, 'images')
        directories = []
        class_names = []
        for filename in os.listdir(image_root):
            path = os.path.join(image_root, filename)
            if os.path.isdir(path):
                directories.append(path)
                class_names.append(filename)

        photo_filenames = []
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                photo_filenames.append(path)

        return photo_filenames, sorted(class_names)

if __name__ == "__main__":
    image_helper = ImageHelper(verbose=False)
    #image = image_helper.cv_read_img_with_abs_path("/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/157/9.png")
    filenames, class_names = image_helper._get_filenames_and_classes("/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/")

    for filename in tqdm(filenames):
        image = image_helper.cv_read_img_with_abs_path(filename)
        cv2.imwrite(filename[:-3]+'jpg', image)
        #print(filename[:-3])
