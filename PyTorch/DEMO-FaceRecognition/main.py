#import os
#import sys
#sys.path.append("")
from data_helper import data_manager

from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from recognition_engine import face_recognition_net as FRN

batch_size = 20

if __name__ == "__main__":
    writer = SummaryWriter('runs/face_rec_net_1')
    
    t_fdl, v_fdl = data_manager.get_data(
        "/DATASETs/Face/Face_SJC/Original_Data/train/",
        "/DATASETs/Face/Face_SJC/Original_Data/valid/",
        comp_list=[
            data_manager.ToGrayscale(),
            data_manager.Rescale((224, 224)),
            data_manager.Normalize(),
            data_manager.ToTensor()],
        bs=batch_size)

    #train_fds.__len__()
    #print(train_fds.__getitem__(10))

    train_fdl = data_manager.WrappedDataLoader(t_fdl, 224, 224, 1)
    valid_fdl = data_manager.WrappedDataLoader(v_fdl, 224, 224, 1)

    model, opt = FRN.get_model(lr=0.01, verbose=False)
    writer.add_graph(model, FRN.torch.randn(20, 1, 224, 224))
    
    model.float()
    model.to(FRN.torch.device("cuda"))
    
    FRN.fit(47, model, FRN.torch.nn.functional.cross_entropy, opt, train_fdl, valid_fdl, writer=writer)
