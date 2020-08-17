# import opencv
import cv2

import torch
# import torch Dataset and DataLoader class
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, utils, transforms

# import matplotlib for image showing
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import os for directory management
import os
import os.path

from random import randrange

import numpy as np
'''
class FaceDataset():
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.face_dataloader = DataLoader(
            datasets.ImageFolder(root=root_dir),
            batch_size=4,
            shuffle=True,
            num_workers=4)
'''

'''
img = cv2.imread("/DATASETs/Face/Face_SJC/Original_Data/129/1.png")
print(type(img))
print(img.shape)

plt.imshow(img[...,::-1]) # BGR to RGB
plt.show()
'''


'''class FaceDataDataLoader:

    def __call__(self, torch_ds, bs, nw):
        return DataLoader(torch_ds, batch_size=bs, num_workers=nw)
'''


def show_batched_images(images_batch, labels_batch):

    plt.imshow(images_batch.numpy().transpose(1, 2, 0))
    #plt.show()
    '''
    labels_batch = labels_batch.tolist()

    converted_labels_batch = []

    key_ref = list(label_ref.keys())
    val_list = list(label_ref.values())

    for label in labels_batch:
        converted_labels_batch.append(int(key_ref[val_list.index(label)]))

    plt.title(converted_labels_batch)
    plt.show()
    '''

class FaceDataDataset(Dataset):
    """ Load face data as Torch Dataset """
    
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        # read directory name
        #self.directories = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

        self.label_reference = {}

        self.directories = {}
        for dir_idx, dir_name in enumerate(os.listdir(root_dir), 1):
            if os.path.isdir(os.path.join(root_dir, dir_name)):
                self.directories.update({dir_idx: os.path.join(root_dir, dir_name)})
                self.label_reference.update({dir_name:dir_idx})

        # total number of images in a directory
        self.tot_img_in_dirs = {}

        self.file_extension = None
        self.transform = transform

    def __len__(self):

        for label, dir_path in self.directories.items():
            img_extension = ['png', 'jpg']

            # save file paths in a directory into a [files] list
            files = [x for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, x)) and x[-3:] in img_extension]
            self.file_extension = files[0][-4:]

            self.tot_img_in_dirs.update({label: len(files)})  # save total image number of a directory

        # return total number of images of all directories
        return sum([self.tot_img_in_dirs[x] for x in self.tot_img_in_dirs])

    def __getitem__(self, idx):
        idx_label = self.convert_idx_to_label(idx)
        #print('idx_label: ', idx_label)
        #print(self.directories)
        #print(self.tot_img_in_dirs)

        img_path = os.path.join(self.directories[idx_label], str(randrange(self.tot_img_in_dirs[idx_label]))+self.file_extension)
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        #plt.show()

        sample = {'image': img, 'label': np.array(idx_label)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def convert_idx_to_label(self, idx):
        idx_label = 0
        label_finder = 0

        for label, tot_num in self.tot_img_in_dirs.items():
            label_finder += tot_num

            if label_finder > idx:
                idx_label = label
                break

        return idx_label
   

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(np.shape(image))

        if len(np.shape(image)) > 2:
            image = image.transpose((2, 0, 1))
        else:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)

        #print(np.shape(image))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class ToGrayscale(object):
    """Convert 3 channels cv ndarray image in sample to Grayscale."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return {'image': image,
                'label': label}


class Rescale(object):
    """Rescale cv ndarray image in sample."""
    def __init__(self, output_dim):
        self.output_dim = output_dim
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = cv2.resize(image, self.output_dim, interpolation=cv2.INTER_AREA)

        return {'image': image,
                'label': label}


class Normalize(object):
    #def __init__(self):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)

        #image = cv2.normalize(image, 0, 255, cv2.NORM_MINMAX)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        #print(np.max(image), np.min(image))

        return {'image': image,
                'label': label}


def get_data(train_dir=None, valid_dir=None, comp_list=[], bs=0):
    train_ds = None
    valid_ds = None

    if valid_dir is not None:
        train_ds = FaceDataDataset(
            train_dir,
            transform=transforms.Compose(comp_list))

    if valid_dir is not None:
        valid_ds = FaceDataDataset(
            valid_dir,
            transform=transforms.Compose(comp_list))

    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs, shuffle=True)
    )


class WrappedDataLoader:
    def __init__(self, dl, w, h, c):
        self.dl = dl
        self.w = w
        self.h = h
        self.c = c

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)

        for b in batches:
            yield (self.img_size_preprocess(b['image'], b['label']))

    def img_size_preprocess(self, x, y):
        x = x.view(-1, self.c, self.w, self.h)#.to(torch.device("cuda"))
        y = y#.to(torch.device("cuda"))
        return x, y
