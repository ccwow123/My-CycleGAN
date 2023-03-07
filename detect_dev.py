#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
'''
由A生成B 缺陷到正常 所以看生成的B文件夹，结果一定处理即可得到缺陷检测结果
由B生成A 正常到缺陷 所以看生成的A文件夹即可得到数据增强
'''
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/cap_b2cap_g', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true',default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output_ori/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output_ori/netG_B2A.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

class Detecter:
    def __init__(self, args):
        self.args =args

    def create_save_path(self):
        save_path = os.path.join('output',self.args.dataroot.split('/')[-1])
        save_path_A = os.path.join(save_path, 'A')
        save_path_B = os.path.join(save_path, 'B')
        if not os.path.exists(save_path_A):
            os.makedirs(save_path_A)
        if not os.path.exists(save_path_B):
            os.makedirs(save_path_B)
        return save_path_A, save_path_B

    def load_dataset(self):
        transforms_ = [transforms.ToTensor(),
                       transforms.Resize((self.args.size, self.args.size)),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = DataLoader(ImageDataset(self.args.dataroot, transforms_=transforms_, mode='test'),
                                batch_size=self.args.batchSize, shuffle=False, num_workers=self.args.n_cpu)
        return dataloader

    def create_models(self):
        # Networks
        netG_A2B = Generator(self.args.input_nc, self.args.output_nc)
        netG_B2A = Generator(self.args.output_nc, self.args.input_nc)
        if self.args.cuda:
            netG_A2B.cuda()
            netG_B2A.cuda()
        # Load state dicts
        netG_A2B.load_state_dict(torch.load(self.args.generator_A2B))
        netG_B2A.load_state_dict(torch.load(self.args.generator_B2A))
        # Set model's test mode
        netG_A2B.eval()
        netG_B2A.eval()
        return netG_A2B, netG_B2A

    def detect_func(self, batch, dataloader, i, input_A, input_B, netG_A2B, netG_B2A, save_path_A, save_path_B):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)#由A生成B 缺陷到正常
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)#由B生成A 正常到缺陷
        # Save image files
        save_image(fake_A, os.path.join(save_path_A, '%04d.png' % (i + 1)))
        save_image(fake_B, os.path.join(save_path_B, '%04d.png' % (i + 1)))
        # sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))


    def run(self):
        ###### Definition of variables ######
        netG_A2B, netG_B2A = self.create_models()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.Tensor
        input_A = Tensor(self.args.batchSize, self.args.input_nc, self.args.size, self.args.size)
        input_B = Tensor(self.args.batchSize, self.args.output_nc, self.args.size, self.args.size)

        # Dataset loader
        dataloader = self.load_dataset()
        ###################################

        ###### Testing######

        # Create output dirs if they don't exist
        save_path_A, save_path_B=self.create_save_path()

        # for i, batch in enumerate(dataloader):
        #     self.method_name(batch, dataloader, i, input_A, input_B, netG_A2B, netG_B2A, save_path_A, save_path_B)
        # sys.stdout.write('\n')
        ###################################
        # 多线程
        pool = ThreadPoolExecutor()
        for i, batch in enumerate(tqdm(dataloader)):
            pool.submit(self.detect_func, batch, dataloader, i, input_A, input_B, netG_A2B, netG_B2A, save_path_A, save_path_B)


if __name__ == '__main__':
    args = parser_args()
    detecter = Detecter(args)
    detecter.run()