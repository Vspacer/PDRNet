import torch
import torch.nn as nn
import cv2
import numpy as np
import os, glob, datetime, time
from torch.autograd import Variable
import torch.nn.init as init
from my_dwt_tensor import dwt_97, idwt_97



class PDRNet_v0(nn.Module):
    def __init__(self, use_borm=True):
        super(PDRNet_v0, self).__init__()
        self.net_conv1 = self.net_conv(in_channels=3, out_channels=128)
        self.net_unit1 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_unit2 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_conv2 = self.net_conv(in_channels=128, out_channels=3)
        self.net_conv3 = self.net_conv(in_channels=6, out_channels=128)
        self.net_unit3 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_unit4 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_conv4 = self.net_conv(in_channels=128, out_channels=3)
        self.net_conv5 = self.net_conv(in_channels=6, out_channels=128)
        self.net_unit5 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_unit6 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_unit7 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.net_unit8 = self.net_unit(in_channels1=128, out_channels1=128, in_channels2=128, out_channels2=128,\
                in_channels3=128, out_channels3=128)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.Prelu1 = nn.PReLU()
        self.Prelu2 = nn.PReLU()
        self.Prelu3 = nn.PReLU()
        self.Prelu4 = nn.PReLU()
        self.Prelu5 = nn.PReLU()
        self.Prelu6 = nn.PReLU()
        self.Prelu7 = nn.PReLU()
        self.Prelu8 = nn.PReLU()

        self._initialize_weights()
    def net_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.0001, momentum = 0.95),
                nn.ReLU())

    def net_unit(self, in_channels1, out_channels1, in_channels2, out_channels2, in_channels3, out_channels3):
        return nn.Sequential(
                self.net_block(in_channels1, out_channels1),
                self.net_block(in_channels2, out_channels2),
                nn.Conv2d(in_channels=in_channels3, out_channels=out_channels3,kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels3, eps=0.0001, momentum = 0.95))

    def net_conv(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.0001, momentum = 0.95))

    def forward(self, x):
        x0 = x # Nx3xWxH
        subband1 = dwt_97(x0) #Nx12xWxH
        subband1 = subband1[:,:3,:,:]
        subband2 = dwt_97(subband1)#Nx12xWxH
        subband2 = subband2[:,:3,:,:]


        x1 = self.net_conv1(subband2)
        out = self.net_unit1(x1)
        out = self.Prelu1(x1+out)
        x2 = out
        out = self.net_unit2(out)
        out = self.Prelu2(x2+out)
        out = self.net_conv2(out)
        y_subband2 = out + subband2
        out = y_subband2


        high_frequency1 = torch.zeros(out.shape, dtype=torch.float32, requires_grad=True).cuda()
        out = torch.cat((out, high_frequency1, high_frequency1, high_frequency1),1)
        out = idwt_97(out) #Nx3xWxH
        #out_tmp = out
        out = torch.cat((subband1, out), 1)
        out = self.net_conv3(out)
        x3 = out
        out = self.net_unit3(out)
        out = self.Prelu3(x3+out)
        x4 = out
        out = self.net_unit4(out)
        out = self.Prelu4(x4+out)
        out = self.net_conv4(out)
        y_subband1 = out + subband1
        out = y_subband1


        high_frequency2 = torch.zeros(out.shape, dtype=torch.float32, requires_grad=True).cuda()
        out = torch.cat((out, high_frequency2, high_frequency2, high_frequency2),1)
        out = idwt_97(out)
        out = torch.cat((out, x0), 1)
        out = self.net_conv5(out)
        x5 = out
        out = self.net_unit5(out)
        out = self.Prelu5(x5+out)
        x6 = out
        out = self.net_unit6(out)
        out = self.Prelu6(x6+out)
        x7 = out
        out = self.net_unit7(out)
        out = self.Prelu7(x7+out)
        x8 = out
        out = self.net_unit8(out)
        out = self.Prelu8(x8+out)
        out = self.conv1(out)

        return out  +  x0, y_subband1, y_subband2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

