"""
This file contains the official implementation of SEFRN (single-coiled version)
"""


import torch
import torch.nn as nn
from .srrfn_model import *
from .networkUtil import *
import pdb

#===============================
# Basic Module
#===============================

def conv1d(in_channels,out_channels):
    conv = nn.Conv2d(in_channels, out_channels,kernel_size=1)
    return conv


class ASIM(nn.Module):
    def __init__(self, in_feat=2, mid_feat=32, M=2, r=16 ,stride=1, L=16):
        """ ASIM Module
        Args:
            in_feat: input channel dimensionality, we have 2 here 
            mid_feat: output channel dimensionality, we have 2 here 
            M: the number of branchs. we have 2 branch 
            r: the reduction radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ASIM, self).__init__()
        self.M = M
        self.in_feat = in_feat
        self.mid_feat = mid_feat
        self.split_convs = nn.ModuleList([])

        for i in range(M): # for each branch
            self.split_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_feat, mid_feat, kernel_size=1, bias=False), # use 1*1 kernel
                        nn.BatchNorm2d(mid_feat),
                        nn.ReLU(inplace=False)
            ))

        d = max(int(mid_feat/r), L) # max(32/16, 16) = 16
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(mid_feat, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
 
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d,mid_feat,kernel_size=1,stride=1) 
            )
        self.softmax = nn.Softmax(dim=1)
        self.tail = nn.Conv2d(mid_feat,in_feat,1)

    def forward(self, x1, x2):
        """
        x1: the first recurrent branch
        x2: the second recurrent branch
        """
       
        batch_size = x1.shape[0]
        feats = []
        # split, conv
        for i, conv in enumerate(self.split_convs): # using 3*3 conv
            if i == 0:
                fea = conv(x1) #(8,32,256,256)
            else:
                fea = conv(x2)
            feats.append(fea)
        
        feas = torch.cat(feats, dim=1) #(8,2,32,256,256)
        feas = feas.view(batch_size, self.M, self.mid_feat, feas.shape[2], feas.shape[3]) # (8,2,32,256,256)
        
        # add the splits
        fea_U = torch.sum(feas, dim=1) # (8,32,256,256)
        fea_S = self.gap(fea_U) #(8,32,1,1)
        fea_Z = self.fc(fea_S) # (8,d,1,1)
        
        attention_vectors = [fc(fea_Z) for fc in self.fcs] # d -> 32 
        attention_vectors = torch.cat(attention_vectors,dim=1) # (8,64,1,1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.mid_feat, 1, 1) # (8,2,32,1,1)
        attention_vectors = self.softmax(attention_vectors) # (8,2,32,1,1)
        
        fea_V = (feas * attention_vectors).sum(dim=1)
        res = self.tail(fea_V)

        return res 




class WAM(nn.Module):
    """
    input: (b,c,h,w)
    output: (b,c,h,w)
    """
    def __init__(self, inFeat=2, midFeat=64):
        super(WAM,self).__init__()
        self.head = nn.Conv2d(2*inFeat,midFeat,kernel_size=1)
        self.act = nn.ReLU()
        self.tail = nn.Conv2d(midFeat, inFeat,kernel_size=1)

    def forward(self,x1,x2):
        """
        x1: (8,2,256,256)
        """
        x = torch.cat([x1,x2],dim=1)
        x = self.head(x) 
        x = self.act(x)
        x = self.tail(x)
        return x


class DCRG(nn.Module):
    """
    DCRG module
    """
    def __init__(self,inChannel=2, fmChannel=64,kernel_size=3, act=nn.ReLU(True), n_resblocks=5, isFastmri=False):
        super(DCRG,self).__init__()

        self.dc1 = dataConsistencyLayer_fastmri(isFastmri=isFastmri)
        self.head = conv1d(inChannel, fmChannel)
        self.dam = ResidualGroup(fmChannel, kernel_size, act=act, res_scale=1, n_resblocks=n_resblocks) 
        self.tail = conv1d(fmChannel, inChannel)
        self.dc2 = dataConsistencyLayer_fastmri(isFastmri=isFastmri)


    def forward(self,x,y,mask):

        x1 = self.dc1(x, y, mask) 
        x1 = self.head(x1)
        x1 = self.dam(x1) 
        x1 = self.tail(x1) # decrease channel
        x2 = self.dc2(x1,y,mask)
        return x2


class WADCRG(nn.Module):
    """
    WADCRG module
    """
    def __init__(self, inChannel = 2, wChannel = 32, fmChannel=64, kernel_size = 3, n_resblocks=5, act= nn.ReLU(True), isFastmri=False):

        super(WADCRG,self).__init__()
        self.wam = WAM(inChannel, wChannel)
        self.dcrg = DCRG(inChannel, fmChannel, kernel_size, act, n_resblocks, isFastmri=isFastmri)
        self.last_hidden = None

    def forward(self, x, y, mask):
       
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x1 = self.wam(x,self.last_hidden) # fuse with prev input
        x2 = self.dcrg(x1, y, mask)
        self.last_hidden = x2
        return x2

    def reset_state(self):
        self.should_reset = True
        self.last_hidden = None


#===============================
# Main Network Architecture
#===============================

class SEFRN(nn.Module):
    def __init__(self, inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 6, kernel_size = 3, n_resblocks=5, act= nn.ReLU(True), isFastmri=False):
        """
        inChannel = 2
        subgroups = 5 
        kernel_size = 3
        n_resblocks = 10
        """
        super(SEFRN, self).__init__()
        self.recur_times = c
        layers = []
        for _ in range(nmodule):
            layers.append(\
                    WADCRG(inChannel=inChannel, \
                          wChannel=wChannel,\
                          fmChannel=fmChannel,\
                          kernel_size = kernel_size,\
                          n_resblocks = n_resblocks, \
                          act = act,\
                          isFastmri = isFastmri) 
                    )

        self.layerList = nn.ModuleList(layers)
        self.skconv = ASIM(inChannel, skChannel, M, r,1, L) 
        self.sktail = dataConsistencyLayer_fastmri(isFastmri=isFastmri)
        self.sktail1 = dataConsistencyLayer_fastmri(isFastmri=isFastmri)
        self.last_hidden = None

    def forward(self, x, y, mask):
    
        self._reset_state()
        x1 = x 
        outs = []

        for ii in range(self.recur_times):
            for layer in self.layerList:
                x1 = layer(x1,y,mask)

            outs.append(x1)
            if ii == 0:
                self.last_hidden = x1
            elif ii == 1:
                x2 = self.skconv(self.last_hidden, x1)
                self.last_hidden = self.sktail(x2,y,mask) # dc consistency
            elif ii == 2:
                x2 = self.skconv(self.last_hidden, x1)
                self.last_hidden = self.sktail1(x2,y,mask) # dc consistency

        outs.append(self.last_hidden) 
        return outs


    def _reset_state(self):
        self.last_hidden = None
        for layer in self.layerList:
            layer.reset_state()








