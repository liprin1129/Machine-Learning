# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import math
import random
import sys

import tensorflow as tf

import cv2
import numpy as np
from tqdm import tqdm

#import matplotlib.pyplot as plt

class ImageHelper(object):
    """
    Helper class that provides TensorFlow image coding utilities.
    
    Args: 
        height: Prefered height of an output image. 
                Default is 15 pixel.
        width:  Prefered width of an output image. 
                Default is 15 pixel.

    Methods:
        cv_read_img_with_abs_path(self, abs_file_name)
        cv_bgra2rgb(self, images)
        cv_bgr2rgb(self, image)
        
    """
    def __init__(self, height=15, width=15, verbose=False):
        self._height = height
        self._width = width
        self._channels = 3
        #self._image_data = None
        self._verbose_img = verbose

    def cv_read_img_with_abs_path(self, abs_file_name):
        image = cv2.imread(abs_file_name, cv2.IMREAD_UNCHANGED)
        if self._verbose_img==True: print("1. Number of channels: ", image.shape); sys.stdout.flush()
        print("Image shape: ", image.shape)

        if image.shape[2] > self._channels: # if image is in RGBD format
            image = self.cv_bgra2rgb(image)
        elif image.shape[2] == self._channels:
            image = self.cv_bgr2rgb(image)
        
        if image.shape[0]!=15 and image.shape[1]!=15: # default image size, 15X15
            image = cv2.resize(image, dsize=(self._width, self._height), interpolation = cv2.INTER_AREA)

        if self._verbose_img==True: print("2. Number of channels: ", image.shape); sys.stdout.flush()
        
        return image.astype(np.float32)

    def cv_bgra2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    def cv_bgr2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class TFRecord_Helper(ImageHelper):
    """
    Helper class which converts image to TFRecord format with features including label, encoded, height, width, format of the image.
    
    Args:
        height:  height of desired output image
        width: width of desired output image
        seed: random seed to shuffle images

    Methods:
        _get_dataset_filename(self, tfrecord_dir, split_name, shard_id, dataset_name, num_shards)
        _get_filenames_and_classes(self, dataset_dir)
        _int64_feature(self, value)
        _bytes_feature(self, value)
        convert_to_tfrecord(self, tf_record_root_dir, dataset_name, num_shards, validation_ratio)
    """
    def __init__(self, height=15, width=15, seed=0, verbose=False):
        ImageHelper.__init__(self, height, width, False)
        self._random_seed = seed
        self._verbose_tfr = verbose

    def _get_dataset_filename(self, tfrecord_dir, split_name, shard_id, dataset_name, num_shards):
        assert split_name in ['train', 'valid']
        
        output_filename = dataset_name + '_%s_%05d-of-%05d.tfrecord' % (
            split_name, shard_id, num_shards)
        return os.path.join(tfrecord_dir, output_filename)

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

    def _int64_feature(self, value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _bytes_feature(self, value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to_tfrecord(self, tf_record_root_dir, dataset_name, num_shards, validation_ratio):
        #print(self._get_dataset_filename(root_image_dir, 'train', 2, 'face', 5))
        #print(self._get_filenames_and_classes(root_image_dir))

        def _seperate_train_and_validation_set():
            """ Inner function of covert_to_tfrecord """
            filenames, class_names = self._get_filenames_and_classes(tf_record_root_dir)
            class_names_to_ids = dict(zip(class_names, range(len(class_names))))
            if self._verbose_tfr==True: print(class_names_to_ids); sys.stdout.flush()

            # Cacluate number of validation proportional to validation_ratio
            num_validation = int(len(filenames) * validation_ratio)

            # Divide into train and validation:
            random.seed(self._random_seed)
            random.shuffle(filenames)
            training_filenames = filenames[num_validation:]
            validation_filenames = filenames[:num_validation]

            return training_filenames, validation_filenames, class_names_to_ids

        train_filenames, valid_filenames, class_names_to_ids = _seperate_train_and_validation_set()
        if self._verbose_tfr==True: print("Number of training-set: {0}, Number of validation-set: {1}".format(len(train_filenames), len(valid_filenames))); sys.stdout.flush()
        
        for filenames, train_or_valid in tqdm(zip((train_filenames, valid_filenames), ('train', 'valid'))):
            num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
            if self._verbose_tfr==True: print("Number per shard: ", num_per_shard); sys.stdout.flush()

            for shard_id in tqdm(range(num_shards)):
                output_filename = self._get_dataset_filename(tf_record_root_dir, train_or_valid, shard_id, dataset_name, num_shards)
                if self._verbose_tfr==True: print(output_filename); sys.stdout.flush()

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        #sys.stdout.write('\r>> Converting [%s] image %d/%d shard %d' % (train_or_valid, i + 1, len(filenames), shard_id))
                        #print('Converting [{0}] image {1}/{2} shard {3}'.format(train_or_valid, i + 1, len(filenames), shard_id))
                        #sys.stdout.flush()

                        # Read the filename:
                        if self._verbose_tfr==True: print(filenames[i]); sys.stdout.flush()
                        #image_data_cv = self.cv_read_img_with_abs_path(filenames[i])
                        #image_data_binary = tf.compat.as_bytes(image_data_cv.tostring())
                        # Read image data in terms of bytes
                        with tf.gfile.FastGFile(filenames[i], 'rb') as fid:
                            image_data_binary = fid.read()
                        #image_data_binary = image_data_cv.tostring()
                        #height, width = image_reader.read_image_dims(sess, image_data)
                        
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        
                        # Get a class ids
                        class_id = class_names_to_ids[class_name]
                        if self._verbose_tfr==True: print(class_id); sys.stdout.flush()

                        # Create a feature
                        feature = {'image/label': self._int64_feature(np.int64(class_id)),
                            'image/encoded': self._bytes_feature(image_data_binary),
                            'image/height': self._int64_feature(np.int64(self._height)),
                            'image/width': self._int64_feature(np.int64(self._width)),
                            'image/channel': self._int64_feature(self._channels)}

                        # Create an Example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=feature))

                        # Serialize to string and write on the file 
                        tfrecord_writer.write(example.SerializeToString())
                        '''
                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
                        '''
        print('\n')

    def convert_from_tfrecord(self, tf_record_root_dir):
        # Create a feature
        feature = {'image/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channel': tf.FixedLenFeature([], tf.int64)}
        
        """feature = {'image/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.VarLenFeature(tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channel': tf.FixedLenFeature([], tf.int64)}"""

        def _get_list():
            """
            Neted function of conver_from_tfrecord.
            Returns:
                filenames: list of file names which have tfrecord file format
            """
            #import time
            #start = time.time()
            """
            filenames = []
            for tfrecord_filename in os.listdir(tf_record_root_dir):
                 if tfrecord_filename.endswith('.tfrecord'):
                     filenames.append(tfrecord_filename)
            """
            filenames = [os.path.abspath(tfrecord_filename) for tfrecord_filename in os.listdir(tf_record_root_dir) if tfrecord_filename.endswith('.tfrecord')]
            #time_difference = time.time() - start
            #print(time_difference)
            return filenames

        tfrecord_filenames = _get_list()
        #if self._verbose_tfr==True: [print(tfrecord_filename) for tfrecord_filename in tfrecord_filenames]; sys.stdout.flush()

# ########################################################################################################
# ########################################################################################################
# ########################################################################################################        
        def _extract_from_tfrecord(_tfrecord_filenames):
            # Extract the data record
            datum_record = tf.parse_single_example(_tfrecord_filenames, feature)

            image = tf.image.decode_image(datum_record['image/encoded'])
            image_shape = tf.stack([datum_record['image/height'], datum_record['image/height'], datum_record['image/channel']])
            label = datum_record['image/label']
            #print("Extracted image shape: ", np.shape(image)); sys.stdout.flush()
            #print("Extracted image shape: ", image_shape); sys.stdout.flush()
            #print(datum_record['image/height'], datum_record['image/height'], datum_record['image/channel']); sys.stdout.flush()

            return image, image_shape, label

        #for train_or_valid in ('train', 'valid'):
        #tfrecord_dataset = tf.data.TFRecordDataset([i for i in tfrecord_filenames if train_or_valid in i][0])
        tfrecord_dataset = tf.data.TFRecordDataset(["/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/face_train_00000-of-00006.tfrecord"])
        #if self._verbose_tfr==True: [print(i) for i in tfrecord_filenames if train_or_valid in i]; sys.stdout.flush()
        
        tfrecord_dataset = tfrecord_dataset.map(_extract_from_tfrecord)
        
        tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
        get_next_in_interator = tfrecord_iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            #image_data = sess.run(get_next_in_interator)
            #print("Extracted image label: ", np.shape(image_data[2])); sys.stdout.flush()

            try:
                # Keep extracting data till TFRecord is exhausted
                while True:
                    image_data = sess.run(get_next_in_interator)

                    """
                    # Check if image shape is same after decoding
                    if not np.array_equal(image_data[0].shape, image_data[3]):
                        print('Image {} not decoded properly'.format(image_data[2]))
                        continue
                        
                    save_path = os.path.abspath(os.path.join(folder_path, image_data[2].decode('utf-8')))
                    mpimg.imsave(save_path, image_data[0])
                    print('Save path = ', save_path, ', Label = ', image_data[1])
                    """
                    print("Extracted image label: ", image_data[2]); sys.stdout.flush()

            except:
                print("ERROR!")
                pass

# ########################################################################################################
# ########################################################################################################
# ########################################################################################################        

if __name__ == "__main__":
    
    select = 1
    #image = image_helper.cv_read_img_with_abs_path("/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/slim/face-recognition/dataset/images/370/370-11.jpg")

    if select == 0:
        image_helper = TFRecord_Helper(height=224, width=224, verbose=False)
        image_helper.convert_to_tfrecord(
            #'/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/slim/face-recognition/dataset/',
            '/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/',
            'face', 6, 0.3)
    
    else:
        image_helper = TFRecord_Helper(height=224, width=224, verbose=True)
        image_helper.convert_from_tfrecord('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/')
    
    #ii = image_helper.tf_read_file_as_binary("/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/170/170-11.png")
    #print(type(ii))
    #with tf.Session('') as sess:
