import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .networkUtil import *


class convBlock(nn.Module):
    def __init__(self, indim=2, iConvNum = 5, f=64):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(indim,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            convList.append(nn.Conv2d(f,f,3,padding = 1))
            convList.append(nn.BatchNorm2d(f))
            convList.append(nn.ReLU())

        self.layerList = nn.ModuleList(convList)

        self.conv2 = nn.Conv2d(f,indim,3,padding = 1)
        self.bn2 = nn.BatchNorm2d(indim)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        for layer in self.layerList:
            x2 = layer(x2)
        
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        
        return x2 + x1

class MoDL(nn.Module):
    def __init__(self, d = 10, c = 3, fNum = 32, isFastmri=False):
        super(MoDL, self).__init__()
        templayerList = []
        self.recur = c
        tmpConv = convBlock(2, d, fNum)
        tmpDF = dataConsistencyLayer_fastmri(isFastmri=isFastmri)
        templayerList.append(tmpConv)
        templayerList.append(tmpDF)

        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xt = x1
        for _ in range(self.recur):
            flag = True
            for layer in self.layerList:
                if(flag):
                    xt = layer(xt)
                    flag = False
                else:
                    xt = layer(xt, y, mask)
                    flag = True
        
        return xt


