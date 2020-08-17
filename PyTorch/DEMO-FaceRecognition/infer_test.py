import torch
from recognition_engine import face_recognition_net as FRN
from data_helper import data_manager
from torchvision import transforms
import numpy as np
import time

from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/face_rec_net_1')


model = FRN.FaceNet(False)
model.load_state_dict(torch.load("./data/test.pt"))
model.to(torch.device("cuda"))
model.eval()


t_fdl, v_fdl = data_manager.get_data(
    "/DATASETs/Face/Face_SJC/Original_Data/train/",
    "/DATASETs/Face/Face_SJC/Original_Data/valid/",
    comp_list=[
        data_manager.ToGrayscale(),
        data_manager.Rescale((224, 224)),
        data_manager.Normalize(),
        data_manager.ToTensor()],
    bs=20)

valid_fdl = data_manager.WrappedDataLoader(v_fdl, 224, 224, 1)
#valid_fdl = next(iter(valid_fdl))

for x, y in valid_fdl:
#x = valid_fdl[0].to(torch.device("cuda"))
#y = valid_fdl[1]

    x = x.cuda()

    #print(x.size())
    #print(model.device, x.device)
    #print(valid_fdl)
    output = model.forward(x)
    #print(output.size())
    #output = torch.nn.functional.softmax(output, dim=0)

    val, indc = torch.max(output, dim=1)

    #print(y.tolist(), '\n', indc.tolist())


    fig = plt.figure(figsize=(12, 48))

    for idx in np.arange(8):
        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
        #print(np.shape(x[idx].cpu().numpy().transpose(1, 2, 0)))
        idx = np.random.randint(20)
        plt.imshow(x[idx].cpu().numpy().squeeze())

        ax.set_title("label: {0}, \n(pred: {1})".format(
            y[idx],
            indc[idx]),
            color=("green" if y[idx] == indc[idx] else "red"))

    '''
    writer.add_figure(
        'actual vs prediction',
        fig)
    writer.close()
    '''

    plt.show()
'''
output_list = output.to(torch.device("cpu")).tolist()

print("Answ:", np.asarray(y.tolist(), dtype=np.int))
print("pred:", np.argmax(output_list, axis=1))
print("prob:", torch.nn.functional.softmax(output, dim=0).size())
'''