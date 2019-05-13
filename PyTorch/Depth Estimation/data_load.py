import numpy as np
from skimage import io
import os

from torch.utils.data import Dataset

import torch

class LoadData(Dataset):
    def __init__(self, root_dir, dataset_list, transform=None):
        """
        Args:
            root_dir (string): Directory with RGB and Depth-TIFF images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.mat_data = h5py.File(root_dir+mat_file, 'r')
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_list = dataset_list
    
    def __len__(self):
        self.dataset_len = len(self.dataset_list)
        return self.dataset_len
    
    def __getitem__(self, idx):
        rgb_img = io.imread(self.root_dir+'Images/RGB/{0:04d}.jpg'.format(self.dataset_list[idx]))
        depth_img = io.imread(self.root_dir+'Images/Depth-TIFF/{0:04d}.tiff'.format(self.dataset_list[idx]))
            
        if self.transform:
            rgb_img = self.transform(rgb_img)
            #depth_img = self.transform(np.expand_dims(depth_img, axis=0))
            depth_img = self.transform(depth_img)
        
        #print(np.max(rgb_img.numpy()), np.min(rgb_img.numpy()))
        #print(np.max(depth_img.numpy()), np.min(depth_img.numpy()))
        sample = {'RGB':rgb_img, 'DEPTH':depth_img}
        
        return sample

# PREPROCESSING
class ToTensor(object):
    def __init__(self, norm=False):
        self.norm = norm
        self.normalizer = Normalization()

    """Convert ndarray in sample to Tensors."""
    def __call__(self, sample_img):
        if len(np.shape(sample_img)) > 2:
            # swap colour axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            
            if self.norm == True:
                sample_img = self.normalizer(sample_img)

            tensor = torch.from_numpy(sample_img.transpose((2, 0, 1)))

            return tensor.float()

        else:
            if self.norm == True:
                sample_img = self.normalizer(sample_img)

            tensor = torch.from_numpy(np.expand_dims(sample_img, axis=0))
            
            return tensor.float()

class Normalization(object):
    """Normalization for a given 2d array"""     
    def __call__(self, sample_img):        
        if len(np.shape(sample_img)) > 2:
            # RGB IMAGE NORMALIZATION W.R.T DIMENSIONS
            rgb_img_norm_ch0 = self._normalizer(sample_img[:, :, 0])
            rgb_img_norm_ch1 = self._normalizer(sample_img[:, :, 1])
            rgb_img_norm_ch2 = self._normalizer(sample_img[:, :, 2])
        
            rgb_img = np.stack((rgb_img_norm_ch0, rgb_img_norm_ch1, rgb_img_norm_ch2))
        
            # swap colour axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            return rgb_img.transpose((1, 2, 0))
        
        else:
            # DEPTH IMAGE NORMALIZATION
            return self._normalizer(sample_img)
        
    def _normalizer(self, array_2d):
        array_scaled = (array_2d - np.min(array_2d)) / (np.max(array_2d) - np.min(array_2d))
        #print(np.max(array_scaled), np.min(array_scaled))
        
        return array_scaled

if __name__ == '__main__':
    def seperate_dataset(testset_ratio=0.3):
        """
        Seperate Train and Test dataset

        Returns: random indice of train test datasets, and length of those.
        """
        dataset_len = len(os.listdir(data_folder+"Images/RGB/"))
        rnd_indice = [i for i in range(dataset_len)]
        np.random.shuffle(rnd_indice)
        split_thresh = int(dataset_len*(1-testset_ratio))

        train_indice = rnd_indice[:split_thresh]
        test_indice = rnd_indice[split_thresh:]

        trainset_len = len(train_indice)
        testset_len = len(test_indice)

        return train_indice, test_indice, trainset_len, testset_len


    from torchvision import transforms

    data_folder = "/home/user170/shared-data/Personal_Dev/Machine-Learning/Data/Depth/NYU-Depth-Dataset-V2/"


    trainset_indice, testset_indice, trainset_len, testset_len = seperate_dataset()

    composed = transforms.Compose([Normalization(), ToTensor()])
    trainset = LoadData(root_dir=data_folder, dataset_list=trainset_indice, transform=composed)

    print("RGB SHAPE: {}, MAX: {}, MIN:{}".format(np.shape(trainset[0]["RGB"].numpy()), np.max(trainset[0]["RGB"].numpy()), np.min(trainset[0]["RGB"].numpy())))
    print("DEPTH SHAPE: {}, MAX: {}, MIN:{}".format(np.shape(trainset[0]["DEPTH"].numpy()), np.max(trainset[0]["DEPTH"].numpy()), np.min(trainset[0]["DEPTH"].numpy())))