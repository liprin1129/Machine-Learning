import numpy as np
from skimage import io
import os

from torch.utils.data import Dataset

import torch

class LoadData(Dataset):
    def __init__(self, root_dir, testset_ratio=0.3, transform=None):
        """
        Args:
            root_dir (string): Directory with RGB and Depth-TIFF images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.mat_data = h5py.File(root_dir+mat_file, 'r')
        self.root_dir = root_dir
        self.transform = transform
        self.testset_ratio = testset_ratio
        
        self.trainset_idx, self.testset_idx, self.trainset_num, self.testset_num = self._seperate_dataset()

    def _seperate_dataset(self):
        """
        Seperate Train and Test dataset
        
        Returns: random indice of train test datasets, and length of those.
        """
        dataset_len = len(os.listdir(self.root_dir+"Images/RGB/"))
        rnd_idx = [i for i in range(dataset_len)]
        np.random.shuffle(rnd_idx)
        split_thresh = int(dataset_len*(1-self.testset_ratio))

        train_idx = rnd_idx[:split_thresh]
        test_idx = rnd_idx[split_thresh:]

        trainset_len = len(train_idx)
        testset_len = len(test_idx)

        return train_idx, test_idx, trainset_len, testset_len
    
    def __len__(self):
        self.dataset_len = len(os.listdir(self.root_dir+"Images/RGB/"))
        return self.dataset_len
    
    def __getitem__(self, idx):
        #rgb_img = Image.open(self.root_dir+'Images/RGB/{0:04d}.jpg'.format(self.trainset_idx[idx]))
        #depth_img = Image.open(self.root_dir+'Images/Depth-TIFF/{0:04d}.tiff'.format(self.trainset_idx[idx]))
        rgb_img = io.imread(self.root_dir+'Images/RGB/{0:04d}.jpg'.format(self.trainset_idx[idx]))
        depth_img = io.imread(self.root_dir+'Images/Depth-TIFF/{0:04d}.tiff'.format(self.trainset_idx[idx]))
            
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)
        
        #print(np.max(rgb_img.numpy()), np.min(rgb_img.numpy()))
        #print(np.max(depth_img.numpy()), np.min(depth_img.numpy()))
        sample = {'RGB':rgb_img, 'DEPTH':depth_img}
        
        return sample

# PREPROCESSING
class ToTensor(object):
    """Convert ndarray in sample to Tensors."""
    def __call__(self, sample_img):
        if len(np.shape(sample_img)) > 2:
            # swap colour axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            return torch.from_numpy(sample_img.transpose((2, 0, 1)))
            #rgb_img = np.moveaxis(np.asarray(rgb_img), [-1], [0])
        else:
            return torch.from_numpy(sample_img)

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
    from torchvision import transforms
    
    data_folder = "/home/user170/shared-data/Personal_Dev/Machine-Learning/Data/Depth/NYU-Depth-Dataset-V2/"

    composed = transforms.Compose([Normalization(), ToTensor()])
    load_data = LoadData(root_dir=data_folder, transform=composed)

    print("RGB SHAPE: {}, MAX: {}, MIN:{}".format(np.shape(load_data[0]["RGB"].numpy()), np.max(load_data[0]["RGB"].numpy()), np.min(load_data[0]["RGB"].numpy())))
    print("DEPTH SHAPE: {}, MAX: {}, MIN:{}".format(np.shape(load_data[0]["DEPTH"].numpy()), np.max(load_data[0]["DEPTH"].numpy()), np.min(load_data[0]["DEPTH"].numpy())))