#!/usr/bin/python3

import argparse
import os
import sys

import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

# from utils.models import Generator
# from utils.datasets import ImageDataset
from utils import *
from utils.datasets import ImageDataset_pix2pix
from utils.models_pix2pix import GeneratorUNet, Discriminator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
'''
由A生成B 缺陷到正常 所以看生成的B文件夹，结果一定处理即可得到缺陷检测结果
由B生成A 正常到缺陷 所以看生成的A文件夹即可得到数据增强
'''
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default=r'D:\Files\_using\good2impurity_patch_samll', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument('--cuda', action='store_true',default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator', type=str, default=r'logs_pix/good2impurity_patch_samll/saved_models/generator_10000.pth', help='A2B generator checkpoint file')
    parser.add_argument('--model_name', default="cycleGAN_ex", choices=['cycleGAN', 'cycleGAN_ex'], help="选择模型")#暂时无用
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

class Detecter:
    def __init__(self, args):
        self.args =args
        self.model_name = args.model_name

    def create_save_path(self):
        # save_path = os.path.join('output',self.args.dataroot.split('/')[-1])
        save_path = os.path.join('output',os.path.split(self.args.dataroot)[-1])
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def load_dataset(self):
        transforms_ = [transforms.ToTensor(),
                       transforms.Resize((self.args.img_height, self.args.img_width), Image.BICUBIC),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = DataLoader(ImageDataset_pix2pix(self.args.dataroot, transforms_=transforms_, mode='testgood'),
                                batch_size=self.args.batchSize, shuffle=False, num_workers=self.args.n_cpu)
        return dataloader

    def create_models(self):
        # Networks
        netG= Generator(self.args.input_nc, self.args.output_nc)
        if self.args.cuda:
            netG.cuda()
        # Load state dicts
        netG.load_state_dict(torch.load(self.args.generator))
        # Set model's test mode
        netG.eval()
        return netG


    def run(self):
        ###### Definition of variables ######
        netG = self.create_models()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.Tensor
        input_A = Tensor(self.args.batchSize, self.args.input_nc,self.args.img_height, self.args.img_width)
        # Dataset loader
        dataloader = self.load_dataset()
        ###################################

        ###### Testing######

        # Create output dirs if they don't exist
        save_path=self.create_save_path()
        print('保存路径：', save_path)
        for i, batch in enumerate(dataloader):
            print('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))

            # Generate output
            fake_B = 0.5*(netG(real_A).data + 1.0)

            # Save image files
            save_image(fake_B, os.path.join(save_path, '%04d.png' % (i + 1)))



if __name__ == '__main__':
    args = parser_args()
    detecter = Detecter(args)
    detecter.run()