from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from recognition_engine import face_recognition_net as FRN
from data_helper import data_manager
from torchvision import utils
import torch

import numpy as np


writer = SummaryWriter('runs/face_rec_net_1')

t_fdl, v_fdl = data_manager.get_data(
    "/DATASETs/Face/Face_SJC/Original_Data/train/",
    "/DATASETs/Face/Face_SJC/Original_Data/valid/",
    comp_list=[
        data_manager.ToGrayscale(),
        data_manager.Rescale((28, 28)), #224
        data_manager.Normalize(),
        data_manager.ToTensor()],
    bs=20)

t_fdl = data_manager.WrappedDataLoader(t_fdl, 28, 28, 1)
v_fdl = data_manager.WrappedDataLoader(v_fdl, 28, 28, 1)

train_diter = iter(t_fdl)
valid_diter = iter(v_fdl)

#######################################
t_imgs, t_lables = next(train_diter)
#print(t_imgs)

img_grid = utils.make_grid(t_imgs)

data_manager.show_batched_images(img_grid, t_lables)

writer.add_image("train_images", img_grid)


########################################
model = FRN.FaceNet(False)
model.load_state_dict(torch.load("../data/test.pt"))
model.to(torch.device("cuda"))

writer.add_graph(model, t_imgs.cuda())
writer.close()

##########################################
# helper function
def select_n_random(data, labels, n=100):

    #Selects n random datapoints and their corresponding labels from a dataset
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


t_imgs, t_lables = next(train_diter)  # first tensor
# select random images and their target indices
for idx, (imgs, lables) in enumerate(train_diter):
    t_imgs = torch.cat([t_imgs, imgs])
    t_lables = torch.cat([t_lables, lables])

    #print(t_imgs.size(), t_lables.size())

    if idx > 5:
        break

#print(torch.randn(160, 28*28).size())
#print(t_lables.size())
#print(t_imgs.size())
writer.add_embedding(torch.randn(160, 28*28), metadata=t_lables, label_img=t_imgs)
writer.close()
