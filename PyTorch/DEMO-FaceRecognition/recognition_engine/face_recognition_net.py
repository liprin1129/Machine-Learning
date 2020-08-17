from torch import nn
import torch
from torch import optim
import numpy as np


class FaceNet(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.conv1 = nn.Conv2d(1, 16, kernel_size=32, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=16, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(48, 12, kernel_size=8, stride=2, padding=1, dilation=1)
        #self.conv4 = nn.Conv2d(144, 12, kernel_size=6, stride=1, padding=0, dilation=1)
        #self.adap_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xb):
        if self.verbose: print(xb.size())

        xb = nn.functional.relu(self.conv1(xb))
        if self.verbose: print(xb.size())

        xb = nn.functional.relu(self.conv2(xb))
        if self.verbose: print(xb.size())

        xb = nn.functional.relu(self.conv3(xb))
        if self.verbose: print(xb.size())

        #xb = nn.functional.avg_pool2d(xb, 3)
        xb = nn.functional.adaptive_avg_pool2d(xb, 1) # AdaptiveAvgPool2d(1)
        if self.verbose: print(xb.size())

        #xb = nn.functional.relu(self.conv4(xb))
        #if self.verbose: print(xb.size())

        #xb = Lambda(lambda xb: xb.view(xb.size(0), -1)).forward(xb)
        xb = xb.view(xb.size(0), -1)
        #xb = torch.tensor(xb, dtype=torch.float32)
        if self.verbose: print(xb.size(), xb.type())

        return xb


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def get_model(lr, verbose):
    model = FaceNet(verbose=verbose)
    return model, optim.SGD(model.parameters(), lr=lr)


def loss_batch(model, loss_func, xb, yb, opt=None):
    yb = torch.tensor(yb, dtype=torch.long).to(torch.device("cuda"))
    #print('======>', xb.device, yb.device)
    #print(model(xb).device, yb.device)
    pred = model(xb)
    #print('======> ', pred.type(), yb.type())
    #print('yb ======> ',yb.type())
    loss = loss_func(pred, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, writer=None):
    img_proj_flag = False
    img_proj = None
    label_proj = None

    for epoch in range(epochs):
        model.train()
        for ti, (xb, yb) in enumerate(train_dl):
            #xb.float().to(torch.device('cuda'))
            #yb.float().to(torch.device('cuda'))
            xb = torch.tensor(xb, dtype=torch.float32).cuda()#to(torch.device("cuda"))
            yb = torch.tensor(yb, dtype=torch.float32).cuda()#to(torch.device("cuda"))

            loss, len_xb = loss_batch(model, loss_func, xb, yb, opt)

            # to project image for tensorboard
            if ti < 1:
                #print(ti, "---- 1")
                img_proj = xb
                label_proj = yb

            if img_proj_flag is False and writer is not None:
                #print(ti, "---- 2")
                img_proj = torch.cat([img_proj, xb])
                label_proj = torch.cat([label_proj, yb])

            if img_proj_flag is False and ti >= 4:
                #print(ti, "---- 3")
                img_proj_flag = True
                writer.add_embedding(
                    torch.randn(img_proj.size(0), img_proj.size(2)*img_proj.size(3)),
                    metadata=label_proj,
                    label_img=img_proj)
                writer.close()

        '''
        #print("---> valid")
        if writer is not None and ti % len_xb*5:
            #print('---> ', running_loss / len_xb*5, epoch * len_xb + ti)
            writer.add_scalar(
                'training loss',
                running_loss / len_xb*5,
                epoch * len_xb + ti)
            writer.close()
            running_loss = 0.0
        '''

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb.cuda(), yb.cuda()) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        writer.add_scalar(
            'validation loss',
            val_loss,
            epoch)

        writer.close()

        print('Val loss: ', epoch, val_loss)

    torch.save(model.state_dict(), './data/test.pt')
    writer.close()

'''
x = torch.rand(10, 1, 224, 224)
y = torch.rand(10).uniform_(0, 12).long()

model, opt = get_model(0.01)
loss_func = nn.functional.cross_entropy
print(loss_func(model(x), y))
'''