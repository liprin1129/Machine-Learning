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
        #self.get_next_in_interators = []

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
                            'image/channel': self._int64_feature(self._channels),
                            'image/name': self._bytes_feature(os.path.basename(filenames[i])[:-4].encode('utf-8'))}

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


    def convert_from_tfrecord(self, tf_record_root_dir, _epoch_num, _batch_size):
        # Create a feature
        feature = {'image/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channel': tf.FixedLenFeature([], tf.int64),
            'image/name': tf.FixedLenFeature([], tf.string)}

        def _get_list():
            """
            Neted function of conver_from_tfrecord.
            Returns:
                filenames: list of file names which have tfrecord file format
            """

            """
            filenames = []
            for tfrecord_filename in os.listdir(tf_record_root_dir):
                 if tfrecord_filename.endswith('.tfrecord'):
                     filenames.append(tfrecord_filename)
            """
            os.chdir(tf_record_root_dir) # Change working directory
            filenames = [os.path.abspath(tfrecord_filename) # Return a list of abolute file name for tfrecords
                for tfrecord_filename in os.listdir(tf_record_root_dir) 
                    if tfrecord_filename.endswith('.tfrecord')] 

            return filenames

        tfrecord_filenames = _get_list() #get tfrecord's absolute file names
        #if self._verbose_tfr==True: [print(tfrecord_filename) for tfrecord_filename in tfrecord_filenames]; sys.stdout.flush()

 
        def _extract_from_tfrecord(_tfrecord_filenames):
            """
            Load tfrecord file on memory
            
            Returns:
                tfrecord iterator generator
            """
            # Extract the data record
            datum_record = tf.parse_single_example(_tfrecord_filenames, feature)

            image = tf.image.decode_image(datum_record['image/encoded'])
            image_shape = tf.stack([datum_record['image/height'], datum_record['image/height'], datum_record['image/channel']])
            image_label = datum_record['image/label']
            image_name = datum_record['image/name']
            return image_name, image_shape, image_label, image

        for train_or_valid in ('train', 'valid'):
            #tfrecord_dataset = tf.data.TFRecordDataset([i for i in tfrecord_filenames if train_or_valid in i])

            for tfrecord_file in [i for i in tfrecord_filenames if train_or_valid in i]:
                #tfrecord_dataset = tf.data.TFRecordDataset(["/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/face_train_00000-of-00006.tfrecord"])
                tfrecord_dataset = tf.data.TFRecordDataset([tfrecord_file])

                #if self._verbose_tfr==True: [print(i) for i in tfrecord_filenames if train_or_valid in i]; sys.stdout.flush()
                
                tfrecord_dataset = tfrecord_dataset.map(_extract_from_tfrecord)

                #tfrecord_dataset = tfrecord_dataset.repeat(_epoch_num).batch(_batch_size)
                
                tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
                get_next_in_interator = tfrecord_iterator.get_next()

                """ # Example of how to use tfrecord_iterator inside this function; convert_from_tfrecord(self, tf_record_root_dir)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    try:
                        # Keep extracting data till TFRecord is exhausted
                        while True:
                            image_data = sess.run(get_next_in_interator)

                            print("Extracted image name: ", image_data[0]); sys.stdout.flush()

                    except:
                        print("End of an {0}".format(os.path.basename(tfrecord_file)))
                        pass
                """

                #get_next_in_interators.append(get_next_in_interator)
                print("[[[[[[[[[[[[ {0} ]]]]]]]]]]]]".format(os.path.basename(tfrecord_file)))
                yield get_next_in_interator


    def convert_from_tfrecord_with_tf_dataset(self, tf_record_root_dir, _batch_size, _phase="test"):
        """ Convert tfrecord files to batch dataset using tensorflow Dataset
        
        Args:
            pahse: String type, "train", "valid", "test. Default is "test"

        Returns:
            iterator: iterator of tensorflow Dataset. Currently, make_one_shot_iterator is being used.
            Future work: update iterator to initializable, reinitializable, and feeadable.
        """

        if _phase == "train": filenames = os.path.join(tf_record_root_dir, "*train*.tfrecord")
        elif _phase == "valid": filenames = os.path.join(tf_record_root_dir, "*valid*.tfrecord")
        elif _phase == "test": filenames = os.path.join(tf_record_root_dir, "*test*.tfrecord")
        
        tf_dataset_files = tf.data.Dataset.list_files(filenames, shuffle=True)

        # parallel fetch tfrecords dataset using the file list in parallel
        tf_dataset = tf_dataset_files.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.TFRecordDataset(filename), cycle_length=os.cpu_count()))

        def _extract_from_tfrecord(_tfrecord_filenames, resize=[]):
            """
            Load tfrecord file on memory
            
            Returns:
                tfrecord iterator generator
            """

            feature = {'image/label': tf.FixedLenFeature([], tf.int64), 'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/height': tf.FixedLenFeature([], tf.int64), 'image/width': tf.FixedLenFeature([], tf.int64),
                        'image/channel': tf.FixedLenFeature([], tf.int64), 'image/name': tf.FixedLenFeature([], tf.string)}

            # Extract the data record
            datum_record = tf.parse_single_example(_tfrecord_filenames, feature)

            jpeg_image = datum_record['image/encoded']
            image = tf.image.decode_jpeg(jpeg_image)
            image = tf.cast(image, tf.float32)
            #image_shape = tf.stack([datum_record['image/height'], datum_record['image/height'], datum_record['image/channel']])
            image_label = datum_record['image/label']
            #image_name = datum_record['image/name']

            assert len(resize) > 1 # Assertion error if image height and width are not defined
            image = tf.image.resize_images(image, resize)#, method=tf.image.ResizeMethod.AREA, align_corners=True)
            #print("===========================> ", image)
            image = image - tf.cast(tf.stack((0, 0, 0)), tf.float32) # to specify the dimension of the tfrecord image. If not using this, it will give an error in makeing model phase

            image = image * (1./255) # Normalization

            #return image_name, image_shape, image_label, image
            #return {"image": image}, {"class_idx": image_label}
            return {"image": image, "labels": image_label}

        # map the parse  function to each example individually in threads*2 parallel calls
        tf_dataset = tf_dataset.map(map_func=lambda example: _extract_from_tfrecord(example, resize=[self._height, self._width]), num_parallel_calls=os.cpu_count())
        
        #All the individually processed examples are then batched and ready for processing
        tf_dataset = tf_dataset.batch(batch_size=_batch_size)
        #tf_dataset = tf_dataset.repeat(_epoch_num).batch(batch_size=_batch_size)

        # Load
        tf_dataset = tf_dataset.prefetch(buffer_size=_batch_size)
        #print("\n====> ", dataset)

        #Input Function
        #iterator = tf_dataset.make_one_shot_iterator()
        iterator = tf_dataset.make_initializable_iterator()

        return iterator


if __name__ == "__main__":
    
    select = 2
    #image = image_helper.cv_read_img_with_abs_path("/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/slim/face-recognition/dataset/images/370/370-11.jpg")

    if select == 0:
        image_helper = TFRecord_Helper(height=224, width=224, verbose=False)
        image_helper.convert_to_tfrecord(
            #'/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/slim/face-recognition/dataset/',
            '/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/',
            'face', 6, 0.3)
    
    elif select == 1:
        image_helper = TFRecord_Helper(height=224, width=224, verbose=True)

        # Example of how to use tfrecord_iterator generator yielded by convert_from_tfrecord(self, tf_record_root_dir)
        get_next_in_interators = image_helper.convert_from_tfrecord('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 2, 10)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Keep extracting data till TFRecord is exhausted
            for id, get_next_in_interator in enumerate(get_next_in_interators):
                print("<<<<<<<<<<<< ITERATOR {0} >>>>>>>>>".format(id))
                try:
                    while True:
                    #print(get_next_in_interator)
                        image_data = sess.run(get_next_in_interator)

                        #print("Extracted image name: ", np.shape(image_data[3])); sys.stdout.flush()
                        #print("Extracted image name: ", image_data[2]); sys.stdout.flush()
                        print("Extracted image name: ", np.shape(image_data)); sys.stdout.flush()
                        break

                except:
                    #print("End of an {0}".format(os.path.basename(tfrecord_file)))
                    print("ERROR")
                    pass
                break

    elif select == 2:
        
        image_helper = TFRecord_Helper(height=224, width=224, verbose=False)
        train_iterator = image_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 10, "train")
        valid_iterator = image_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 10, "valid")
        """
        placeholder_X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        placeholder_y = tf.placeholder(tf.int32, shape = [None])

        # Create separate Datasets for training, validation and testing
        train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
        def map_fn(x, y):
            # Do transformations here
            return x, y
        train_dataset = train_dataset.batch(10).map(lambda x, y: map_fn(x, y))

        print(train_dataset.output_types, train_dataset.output_shapes)
        """
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, (tf.float32, tf.int64), ([None, 224, 224, 3], [None]))
        next_element = iterator.get_next()

        with tf.Session() as sess:
            # Return handles that can be feed as the iterator in sess.run
            training_handle = sess.run(train_iterator.string_handle())
            validation_handle = sess.run(valid_iterator.string_handle())

            for _ in range(3):
                try:
                    count = 0
                    sess.run(train_iterator.initializer)
                    while True:
                        a = sess.run(next_element, feed_dict={handle: training_handle})
                        count += 1
                except tf.errors.OutOfRangeError:
                    print(count)
                    pass

                try:
                    count = 0
                    sess.run(valid_iterator.initializer)
                    while True:
                        a = sess.run(next_element, feed_dict={handle: validation_handle})
                        count += 1
                except tf.errors.OutOfRangeError:
                    print(count)
                    print("\n")
                    pass

    elif select == 3:
        image_helper = TFRecord_Helper(height=224, width=224, verbose=False)
        input_fn = image_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 0, 20, "train")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                while True:
                #print(get_next_in_interator)
                    image_data = sess.run(input_fn)

                    #print("Extracted image name: ", np.shape(image_data[3])); sys.stdout.flush()
                    #print("Extracted image name: ", image_data[2]); sys.stdout.flush()
                    print("Extracted image name: ", np.shape(image_data[0]['image'])); sys.stdout.flush()
                    break

            except:
                #print("End of an {0}".format(os.path.basename(tfrecord_file)))
                print("ERROR")
                pass