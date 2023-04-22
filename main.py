"""
This is the main file for training and evaluation
"""

import os
import sys
import time
import cv2
import logging
from tqdm import tqdm
import pathlib
import argparse
import shutil
import random
import numpy as np
import torch
from torch.nn import functional as F
from network import getNet, getLoss, getOptimizer 
from util import paramNumber 
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import getDataloader, fastmri_format, handle_output
from model_test_funcs import test_save_result_per_slice, test_save_result_per_volume 
import warnings
warnings.filterwarnings("ignore")
import pdb


def create_logger(args, mode):
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    filename = os.path.join(args.exp_dir, "train.log")
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s', 
                        filemode=mode,
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def train_epoch(args, epoch, model, data_loader, optimizer, logger):
    
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
    
        optimizer.zero_grad()
        input, target, subF, mask_var, _, _, _, _, _ = data

        input = input.to(args.device, dtype=torch.float)
        target = target.to(args.device, dtype=torch.float)
        subF = subF.to(args.device, dtype=torch.float)
        mask_var = mask_var.to(args.device,dtype=torch.float)
        output = model(input, subF, mask_var)
    
        if not isinstance(output, list): 
            output = fastmri_format(output)
            loss = F.l1_loss(output, target)
        else:
            loss = 0.
            for _, subModel in enumerate(output):
                subModel = fastmri_format(subModel)
                loss += F.l1_loss(subModel, target)

        loss.backward()
        optimizer.step()
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

        if iter % args.report_interval == 0:
            logger.debug(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()


    return avg_loss, time.perf_counter() - start_epoch



def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')



def load_model(args, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer



def build_model(args):
    model = getNet(args.netType).to(args.device)
    return model


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay = 1e-7) 
    return optimizer


def visualize(args, model, data_loader):
    """
    to be finished
    """
    print("visualizing")
    model.eval()
    jump = 1
    with torch.no_grad():
        for iter, data in tqdm(enumerate(data_loader)):
            if iter % jump == 0:  
                input, target, subF, mask_var, mean, std, maxval, fname, slice = data

                input = input.to(args.device, dtype=torch.float)
                target = target.to(args.device, dtype=torch.float)
                subF = subF.to(args.device, dtype=torch.float)
                mask_var = mask_var.to(args.device,dtype=torch.float)
                mean = mean.unsqueeze(1).unsqueeze(2).to(args.device, dtype=torch.float)
                std = std.unsqueeze(1).unsqueeze(2).to(args.device, dtype=torch.float)

                output = model(input, subF, mask_var)
                output = handle_output(output, 'test')
                
                if args.dataName == 'fastmri':
                    if args.dataMode == 'complex':
                        input = fastmri_format(input) /1e6
                        output =  fastmri_format(output) /1e6
                        target = target /1e6
                    elif args.dataMode == 'real':
                        input = input * std + mean
                        output =  fastmri_format(output) * std + mean
                        target = target * std + mean

                elif args.dataName == 'cc359':
                    if args.dataMode == 'complex':
                        input = fastmri_format(input) * 1e5
                        output =  fastmri_format(output) * 1e5
                        target = target * 1e5

                elif args.dataName == 'cardiac':
                    if args.dataMode == 'complex':
                        input = fastmri_format(input)
                        output =  fastmri_format(output)
                        target = target 

                else:
                    raise NotImplementedError('Please provide correct dataset name: fastmri or cc359')

                target_np = target.detach().cpu().data.numpy()
                output_np = output.detach().cpu().data.numpy()
                input_np = input.detach().cpu().data.numpy()
                mask_np = mask_var.detach().cpu().data.numpy()  #(B,1,W,1)
                temp_shape = mask_np.shape
                if len(temp_shape) == 5: #multi-coil:
                    mask_np = mask_np[:,0,:,:,:]
                    temp_shape = mask_np.shape

                temp = np.ones((temp_shape[0], temp_shape[2], temp_shape[2], 1))
                temp = temp * mask_np
                res_np = 5 * (np.abs(target_np - output_np) / target_np.max())
                zim_res_np = 5 * (np.abs(target_np - input_np) / target_np.max())

                N = len(target_np)
                for idx in range(N):
                    plt.imsave(os.path.join(args.im_root, '{}-{}_gt.png'.format(fname[idx].split('.')[0], slice[idx])), target_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_pred.png'.format(fname[idx].split('.')[0], slice[idx])), output_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_zf.png'.format(fname[idx].split('.')[0], slice[idx])), input_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_res.png'.format(fname[idx].split('.')[0], slice[idx])), res_np[idx], cmap='viridis')
                    plt.imsave(os.path.join(args.im_root, '{}-{}_zim_res.png'.format(fname[idx].split('.')[0], slice[idx])), zim_res_np[idx], cmap='viridis')

                    plt.imsave(os.path.join(args.im_root, '{}-{}_mask.png'.format(fname[idx].split('.')[0], slice[idx])), temp[idx,:,:,0], cmap='gray')
                         



def main(args, is_evaluate=0):

    # create folder
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.im_root = os.path.join(args.exp_dir, 'images')
    if not os.path.exists(args.im_root):
        os.mkdir(args.im_root)

    if (args.resume == 1) or (is_evaluate == 1): 
        logger = create_logger(args, 'a')
        logger.debug("loading model. Resume: {}, Evaluate: {}".format(args.resume, is_evaluate))
        checkpoint, model, optimizer = load_model(args, os.path.join(args.exp_dir, 'model.pt'))

        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        assert start_epoch <= args.num_epochs, "model already finish training, do not resume"
        del checkpoint

    else:
        logger = create_logger(args, 'w')
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
        logger.debug(args)
        logger.debug(model)

    param = paramNumber(model)
    logger.debug("model parameters : {}".format(param))

    # dataloader
    train_loader, dev_loader = getDataloader(args.dataName, args.dataMode, args.batchSize, [args.center_fractions], [args.accer], args.resolution, args.train_root, args.valid_root, args.sample_rate, args.challenge, args.mask_type)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    # training mode
    if not is_evaluate:
        logger.debug("start training")
        for epoch in range(start_epoch, args.num_epochs):
            scheduler.step(epoch)
            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, logger)
            if args.test_by_volume:
                dev_loss, dev_psnr, dev_ssim ,dev_time = test_save_result_per_volume(model, dev_loader, args)
            else:
                dev_loss, dev_psnr, dev_ssim ,dev_time = test_save_result_per_slice(model, dev_loader, args)

            is_new_best = dev_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, dev_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logger.debug(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'DevLoss = {dev_loss:.4g} DevPSNR = {dev_psnr:.4g} DevSSIM = {dev_ssim:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
   
    # evaluating mode
    else:
        logger.debug("Start evaluating (without training)")
        if args.test_by_volume:
            dev_loss, dev_psnr, dev_ssim ,dev_time = test_save_result_per_volume(model, dev_loader, args)
        else:
            dev_loss, dev_psnr, dev_ssim ,dev_time = test_save_result_per_slice(model, dev_loader, args)
        
        logger.debug(f'Epoch = [{start_epoch:4d}] DevLoss = {dev_loss:.4g} DevPSNR = {dev_psnr:.4g} DevSSIM = {dev_ssim:.4g} DevTime = {dev_time:.4f}s')
        visualize(args , model, dev_loader)
 

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43) 
    
    parser.add_argument('--netType', type=str) 
    parser.add_argument('--mask_type', type=str, default='cartesian') 
    parser.add_argument('--train_root', type=str, help='path to store the train data', default='/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/')
    parser.add_argument('--valid_root', type=str, help='path to store the train data', default='/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/')
    parser.add_argument('--dataName', type=str, help='name of the dataset. fastmri/cc359', default='fastmri')
    parser.add_argument('--dataMode', type=str, help="data mode for input data, real/complex", default='complex') 
    parser.add_argument('--challenge', type=str, help='challenge. singlecoil/multicoil', default='singlecoil')
    parser.add_argument('--resolution', type=int, help="resolution of data. 320 for fastmri or 256 for cc359", default=320)
    parser.add_argument('--accer', type=int, default=4) 
    parser.add_argument('--center_fractions', type=float, default=0.08, help='if accer=4, center_fractions should be 0.08; if accer=8, center_fractions should be 0.04. This is the routine from fastmri') 
    parser.add_argument('--exp_dir', type=pathlib.Path, default='./results/', help='Path to store the results')
    

    parser.add_argument('--batchSize', type=int, default=16) 
    parser.add_argument('--num-epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--sample_rate', type=float, help="Sample rate", default=1.)
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=20, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')


    parser.add_argument('--report_interval', type=int, default=100) 
    parser.add_argument('--resume', type=int, default=0, help="resume training") 
    parser.add_argument('--data_parallel', action='store_true',default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')

    parser.add_argument('--test_by_volume', type=int, default=1, help='During testing, whether calculate metrics over volume or per slice') 
    
    return parser.parse_args()


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # for easy
    if args.dataName != 'cardiac':
        if args.accer == 4:
            args.center_fractions = 0.08
        elif args.accer == 8:
            args.center_fractions = 0.04
    else:
        args.accer = 15
    
    # be careful of the is_evaluate param here! if 0, do training, if 1, do evaluation
    main(args, is_evaluate=0)
