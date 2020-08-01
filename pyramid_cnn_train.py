# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html
#

# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.image as img
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from skimage import color as skco
from pyramid_wavelet97_cnn import PDRNet_v0
from my_dwt_tensor import dwt_97, idwt_97

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Params
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch PDRNet')
    parser.add_argument('--model', default='PDRNet_v0', type=str, help='choose a type of model')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--train_data', default='../rainy_image_dataset/rain_light_100/train/rain/', type=str, help='path of train data')
    parser.add_argument('--train_label', default='../rainy_image_dataset/rain_light_100/train/norain/', type=str, help='path of train gt')
    parser.add_argument('--input_size', default=120, type=int, help='size of input')
    parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
    parser.add_argument('--lr', default=0.2*1e-3, type=float, help='initial learning rate for Adam')
    return parser.parse_args()

args = get_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
input_size = args.input_size
input_path = args.train_data
gt_path = args.train_label
num_channel = 3
save_dir = os.path.join('pyramid_models', args.model)
_f_ = open('tmp_log.txt', 'w')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)



class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

import re
pattern = re.compile('(x|_)[0-9]*\.')

def read_data(input_path, gt_path, input_size, num_channel, batch_size):
    input_files= os.listdir(input_path)
    gt_files= os.listdir(gt_path)


    Data  = np.zeros((batch_size, input_size, input_size, num_channel))
    Label = np.zeros((batch_size, input_size, input_size, num_channel))

    for i in range(batch_size):

        r_idx = random.randint(0,len(input_files)-1)

        input_file = input_files[r_idx]
        gt_file = pattern.sub('.', input_file)
        #print(input_file)
        #print(output_file)
        rainy = img.imread(os.path.join(input_path, input_file))
        label = img.imread(os.path.join(gt_path, gt_file))

        if rainy.dtype == "float32":
            rainy = (rainy * 255).astype("uint8")
        if label.dtype == "float32":
            label = (label * 255).astype("uint8")

        x = random.randint(0,rainy.shape[0] - input_size)
        y = random.randint(0,rainy.shape[1] - input_size)

        subim_input = rainy[x : x+input_size, y : y+input_size, :]
        subim_label = label[x : x+input_size, y : y+input_size, :]

        #subim_input = skco.rgb2ycbcr(subim_input)
        #subim_label = skco.rgb2ycbcr(subim_label)

        Data[i,:,:,:] = subim_input
        Label[i,:,:,:] = subim_label


    return Data, Label  #NxHxWxC

def read_data_DID(input_path, gt_path, input_size, num_channel, batch_size):
    input_files= os.listdir(input_path)
    #gt_files= os.listdir(gt_path)


    Data  = np.zeros((batch_size, input_size, input_size, num_channel))
    Label = np.zeros((batch_size, input_size, input_size, num_channel))

    for i in range(batch_size):

        r_idx = random.randint(0,len(input_files)-1)

        input_file = input_files[r_idx]
        #print(input_file)
        #print(output_file)

        input_file = img.imread(input_path + input_file)
        [H, W, C] = input_file.shape
        rainy = input_file[:H, :W//2,:]
        label = input_file[:H,W//2:W,:]

        x = random.randint(0,rainy.shape[0] - input_size)
        y = random.randint(0,rainy.shape[1] - input_size)

        subim_input = rainy[x : x+input_size, y : y+input_size, :]
        subim_label = label[x : x+input_size, y : y+input_size, :]

        #subim_input = skco.rgb2ycbcr(subim_input)
        #subim_label = skco.rgb2ycbcr(subim_label)

        Data[i,:,:,:] = subim_input
        Label[i,:,:,:] = subim_label


    return Data, Label  #NxHxWxC

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def cal_psnr(x_, x):
    mse = ((x_.astype(np.float)-x.astype(np.float))**2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = PDRNet_v0()


    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))

    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    model.train()
    if cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40, 50, 60, 70], gamma=0.5)  # learning rates

    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        epoch_loss = 0
        epoch_psnr = 0
        batch_num = 4000
        start_time = time.time()


        for batch_id in range(batch_num):
            optimizer.zero_grad()
            batch_y, batch_x = read_data(input_path, gt_path, input_size, num_channel, batch_size)
            batch_x = torch.from_numpy(batch_x.transpose(0,3,1,2))
            batch_y = torch.from_numpy(batch_y.transpose(0,3,1,2))#NXCXHXW
            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            if cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            subband1 = dwt_97(batch_x)
            subband1 = subband1[:,:3,:,:]
            subband2 = dwt_97(subband1)
            subband2 = subband2[:,:3,:,:]
            batch_y, y_subband1, y_subband2 = model(batch_y)
            loss = criterion(batch_y, batch_x) + 0.1*criterion(y_subband1, subband1) + 0.1*criterion(y_subband2, subband2)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            tmp_batch_x = batch_x.cpu()
            tmp_batch_y = batch_y.cpu()
            tmp_batch_x = tmp_batch_x.detach().numpy()
            tmp_batch_y = tmp_batch_y.detach().numpy()
            psnr = cal_psnr(tmp_batch_x, tmp_batch_y)
            epoch_psnr += psnr

            if batch_id % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f psnr = %.2f' % (epoch+1, batch_id, batch_num, loss.item()/batch_size, psnr))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f ,  psnr = %.2f , time = %4.2f s' % (epoch+1, epoch_loss/batch_num, epoch_psnr/batch_num, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/batch_num, elapsed_time)), fmt='%2.4f')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))






