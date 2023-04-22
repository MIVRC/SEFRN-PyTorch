import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from .networkUtil import *
from .CNN import Unet_dc, Unet_dc_multicoil 
from .RDN_complex import RDN_complex, RDN_multicoil
from .DC_CNN import DC_CNN, DC_CNN_multicoil
from .cascadeNetwork import CN_Dense
from .SEFRN import SEFRN
from .SEFRN_multicoil import SEFRN_multicoil
from .md_recon import MRIReconstruction as mdr
from .md_recon_multicoil import MRIReconstruction_multicoil as mdr_multicoil 
from .MoDL import MoDL



def getScheduler(optimizer, config):

    schedulerType = config['train']['scheduler']
    if(schedulerType == 'lr'):
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=int(config['train']['lr_scheduler_stepsize']))
    else:
        assert False,"Wrong scheduler type"
        
    return scheduler 


def getOptimizer(param, optimizerType, LR, weightDecay = 0):

    if(optimizerType == 'RMSprop'):
        optimizer = torch.optim.RMSprop(param, lr=LR)
    elif(optimizerType == 'Adam'):
        optimizer = torch.optim.Adam(param, lr=LR, weight_decay = 1e-7) #weight decay for DC_CNN
    elif(optimizerType == 'SGD'):
        optimizer = torch.optim.SGD(param, lr=LR, momentum = 0.9) #sgd + momentum
    else:
        assert False,"Wrong optimizer type"
    return optimizer

# main function to getNet
def getNet(netType):

    #===========DC_CNN============
    if(netType == 'DCCNN'):
        return DC_CNN(isFastmri=False)
    elif(netType == 'DCCNN_fastmri'):
        return DC_CNN(isFastmri=True)
    elif(netType == 'DCCNN_fastmri_multicoil'):
        return DC_CNN_multicoil(indim=30, fNum=96, isFastmri=True)
  

    #===========MoDL============
    elif(netType == 'MoDL'):
        return MoDL(d=5, c=5, fNum=64, isFastmri=False)
    elif(netType == 'MoDL_fastmri'):
        return MoDL(d=10, c=1, fNum=32, isFastmri=True)
    
    #===========RDN===============
    elif(netType == 'RDN_complex_DC'):
        return RDN_complex(dcLayer = 'DC', isFastmri=False)
    elif(netType == 'RDN_complex_DC_fastmri'):
        return RDN_complex(dcLayer = 'DC', isFastmri=True)
    elif(netType == 'RDN_multicoil'):
        return RDN_multicoil(xin1=30,midChannel=96,isFastmri=True)

    #===========cascade===========
    elif(netType == 'cddntdc'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 2, isFastmri=False)
    elif(netType == 'cddntdc_fastmri'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 2, isFastmri=True)

    #===========SEFRN===========
    elif(netType == 'SEFRN'):
        return SEFRN(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=False) 
    elif(netType == 'SEFRN_fastmri'):
        return SEFRN(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=True) 

    #===========Unet============
    elif(netType == 'vanillaCNN'):
        return vanillaCNN()
    elif(netType == 'Unet_dc'):
        return Unet_dc(isFastmri=False)
    elif(netType == 'Unet_dc_fastmri'):
        return Unet_dc(isFastmri=True)
    elif(netType == 'Unet_fastmri_real'):
        return UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
    elif(netType == 'Unet_dc_fastmri_multicoil'):
        return Unet_dc_multicoil(indim=30, isFastmri=True)


    #===========mdr============
    elif (netType == 'mdr'):
        return mdr(isFastmri=False)
    elif (netType == 'mdr_fastmri'):
        return mdr(isFastmri=True)
    elif (netType == 'mdr_fastmri_multicoil'):
        return mdr_multicoil(indim=30, middim=64, isFastmri=True, isMulticoil=True)


    else:
        assert False,"Wrong net type"


def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"



