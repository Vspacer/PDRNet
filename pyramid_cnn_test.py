# -*- coding: utf-8 -*-

# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import matplotlib.image as img
import matplotlib.pyplot as plt
import skimage.color as skco
import cv2
import re
from torchvision import transforms
from pyramid_wavelet97_cnn import PDRNet_v0
from my_dwt_tensor import dwt_97, idwt_97

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./rainy_image_dataset/rain_heavy_100/test/Rain100H/rain/', type=str, help='directory of test dataset')
    parser.add_argument('--gt_dir', default='rainy_image_dataset/rain_heavy_100/test/Rain100H/norain/', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Rain100H'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default='./MWCNN_derain_models_nonovercomplete/rain100H_240_nobn_v2_05_05_1/', type =str,help='directory of the model')
    parser.add_argument('--model_name', default='model_180.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_nim', default=0, type=int, help='save the noisy image, 1 or 0')
    parser.add_argument('--nim_dir', default='data/test/noisy_set12_15', type=str, help='directory for noisy images')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--exist_semi', default=0, type=int,  help='test appointed noisy images, 1 or 0')
    parser.add_argument('--semi_dir', default='Noisy_urban100_sigma25', type=str, help='directory for semi images')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        result = np.clip(result, 0, 255)
        #print(result)
        im = Image.fromarray(result.astype('uint8'))
        im.save(path, 'png')
        #imsave(path, np.clip(result, 0, 1))

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())

if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')


    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0,1]).cuda()
        print('parallel')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    with torch.no_grad():
        for set_cur in args.set_names:

            if not os.path.exists(os.path.join(args.result_dir, set_cur)):
                os.mkdir(os.path.join(args.result_dir, set_cur))
            psnrs = []
            ssims = []
            input_files = os.listdir(args.input_dir)
            pattern = re.compile('(x|_)(in|[0-9])*\.')
            for idx in range(len(input_files)):
                input_file = input_files[idx]
                rainy = img.imread(os.path.join(args.input_dir, input_file)) #rainy HWC
                if(rainy.shape[2]==4):
                    rainy = rainy[:,:,:3]
                print(input_file)
                gt_file = pattern.sub('.', input_file)
                gt = img.imread(os.path.join(args.gt_dir, gt_file))  #ground truth HWC
                print(rainy.shape)
                if rainy.dtype == 'float32':
                    rainy = (rainy * 255).astype("uint8")
                if gt.dtype == 'float32':
                    gt = (gt * 255).astype("uint8")

                start_time = time.time()
                flag = False
                h_mod = rainy.shape[0]%4
                w_mod = rainy.shape[1]%4
                #print(w_mod)
                if h_mod != 0:
                    h_pad = 4-h_mod
                else:
                    h_pad = 0
                if w_mod != 0:
                    w_pad = 4-w_mod
                else:
                    w_pad = 0
                print(w_pad)
                if h_mod != 0 or w_mod != 0:
                    flag = True
                    rainy = np.pad(rainy, ((0,h_pad), (0, w_pad), (0,0)), 'symmetric')
                print(rainy.shape)
                [tmp_h, tmp_w, tmp_c] = rainy.shape
                tmp_rainy = rainy.transpose((2,0,1))
                tmp_rainy = torch.from_numpy(tmp_rainy.astype(np.float32)).view(1, 3, tmp_h, tmp_w).cuda()
                ot, ot_subband1, ot_subband2 = model(tmp_rainy)
                ot = ot.view(3, rainy.shape[0], rainy.shape[1])
                ot = ot.cpu()
                ot = ot.detach().numpy().astype(np.float32)
                ot = ot.transpose((1, 2, 0)) #HWC
                ot = ot[:gt.shape[0], :gt.shape[1], :]

                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

                #norm_range(ot, None)
                if args.save_result:
                    save_result(ot, path=os.path.join(args.result_dir, set_cur, 'ot0_derained' + input_file))  # save the denoised image
                ot = Image.fromarray(np.clip(np.around(ot), 0, 255).astype(np.uint8))
                gt = Image.fromarray(np.clip(np.around(gt), 0, 255).astype(np.uint8))
                ot_y = np.array(ot.convert('YCbCr'))[:,:,0]
                gt_y = np.array(gt.convert('YCbCr'))[:,:,0]
                print(ot_y.shape)
                print(gt_y.shape)
                psnr_x_ = compare_psnr(ot_y, gt_y)

                ssim_x_ = compare_ssim(ot_y, gt_y)
                print('%10s : %10s : %2.4f second PSNR:%.2f  SSIM:%.2f' % (set_cur, input_file, elapsed_time, psnr_x_, ssim_x_))
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
            psnr_avg = np.mean(psnrs)
            ssim_avg = np.mean(ssims)
            if args.save_result:
                save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
            log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))







