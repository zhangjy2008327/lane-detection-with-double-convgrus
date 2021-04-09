import time
import config
import torch.nn as nn
from torchvision import models
from utils import *
from ConvGRUCell import ConvGRU
from ConvGRUCell3 import ConvGRU3

class UNet_TwoConvGRU(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_TwoConvGRU, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 48)
        self.down2 = down(96, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(112, 48)
        self.up4 = up(80, 16)
        self.outc = outconv(16, n_classes)
        # self.convlstm = ConvLSTM(input_size=(8, 16),
        #                          input_dim=512,
        #                          hidden_dim=[512, 512],
        #                          kernel_size=(3, 3),
        #                          num_layers=2,
        #                          batch_first=False,
        #                          bias=True,
        #                          return_all_layers=False)
        self.convlstm = ConvGRU(input_size=(8, 16),
                                 input_dim=128,
                                 hidden_dim=[128, 128],
                                 #hidden_dim=[512],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
        # # self.convlstm2 = ConvGRU2(input_size=(128, 256),
        # #                         input_dim=64,
        # #                         hidden_dim=[64, 64],
        # #                         kernel_size=(1, 1),
        # #                         num_layers=2,
        # #                         batch_first=False,
        # #                         bias=True,
        # #                         return_all_layers=False)
        self.convlstm3 = ConvGRU3(input_size=(64, 128),
                                  input_dim=48,
                                  #hidden_dim=[128],
                                  hidden_dim=[48, 48],
                                  kernel_size=(3, 3),
                                  num_layers=2,
                                  batch_first=False,
                                  bias=True,
                                  return_all_layers=False)
        # # self.convlstm4 = ConvGRU4(input_size=(32, 64),
        # #                           input_dim=256,
        # #                           hidden_dim=[256, 256],
        # #                           kernel_size=(1, 1),
        # #                           num_layers=2,
        # #                           batch_first=False,
        # #                           bias=True,
        # #                           return_all_layers=False)
        # # self.convlstm5 = ConvGRU5(input_size=(16, 32),
        # #                           input_dim=512,
        # #                           hidden_dim=[512, 512],
        # #                           kernel_size=(1, 1),
        # #                           num_layers=2,
        # #                           batch_first=False,
        # #                           bias=True,
        # #                           return_all_layers=False)
    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            ##---------gru---2-------------------
            t2 = x2.unsqueeze(0)
            lstm2, _ = self.convlstm3(t2)
            test2 = lstm2[0][-1,:, :, :, :]
            x21 = torch.cat((x2, test2), dim=1)
            x3 = self.down2(x21)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        # # print('data---', data.shape)
        # # data = [5, 3, 512, 8, 16]
        lstm, _ = self.convlstm(data)
        #
        test = lstm[0][-1, :, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
