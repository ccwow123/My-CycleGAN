#!/usr/bin/python3

import argparse
import os
import shutil
import sys

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import cv2

# from utils.models import Generator
# from utils.datasets import ImageDataset

from utils import Generator,Discriminator,weights_init_normal
from utils.datasets import ImageDataset_pix2pix
from utils.models_pix2pix import GeneratorUNet as Generator_pix2pix , Discriminator as Discriminator_pix2pix ,Discriminator2
from pytorch_fid import fid_score

from train_pix2pix import craete_model
'''
由A生成B 缺陷到正常 所以看生成的B文件夹，结果一定处理即可得到缺陷检测结果
由B生成A 正常到缺陷 所以看生成的A文件夹即可得到数据增强
'''
def parser_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', type=str, default=r'E:\datasets\_using\pix_MLCCn6', help='填入-测试数据集-路径')
    parser.add_argument('--dataroot', type=str, default=r'E:\datasets\_using\Good2MLCCn6', help='填入-测试数据集-路径')
    parser.add_argument('--generator', type=str, default=r'logs_pix/ori-ori-pix_MLCCn6/saved_models/generator_500.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--dataset_mode', type=str, default='testgood',choices=['train','test','testgood'], help='选择数据集模式')
    parser.add_argument('--metric', default=True, help='是否评价生成的图片')
    parser.add_argument('--save_imgs', default=True, help='是否保存生成的图片')

    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument('--cuda', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

class Detecter:
    def __init__(self, args):
        self.args =args

    def create_save_path(self):
        # save_path = os.path.join('output',self.args.dataroot.split('/')[-1])
        save_path = os.path.join('output',os.path.split(self.args.dataroot)[-1]+
                                 '-'+self.args.Discriminator+'-'+self.args.Generator )
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def load_dataset(self):
        transforms_ = [transforms.ToTensor(),
                       transforms.Resize((self.args.img_height, self.args.img_width), Image.BICUBIC),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = DataLoader(ImageDataset_pix2pix(self.args.dataroot, transforms_=transforms_, mode=self.args.dataset_mode),
                                batch_size=self.args.batchSize, shuffle=False, num_workers=self.args.n_cpu)
        return dataloader

    def create_models(self):
        # Networks
        # netG= Generator(self.args.input_nc, self.args.output_nc)
        # 分离生成器和判别器的类型
        self.args.Discriminator =self.args.generator.split('/'or'\\')[1].split('-')[0]
        self.args.Generator = self.args.generator.split('/'or'\\')[1].split('-')[1]
        netG , _= craete_model(self.args)
        if self.args.cuda:
            netG.cuda()
        # Load state dicts
        netG.load_state_dict(torch.load(self.args.generator))
        # Set model's test mode
        netG.eval()
        return netG
    



    def metric_func_FID(self, save_path):
        # 定义真实图像和生成图像的路径
        real_path = os.path.join(save_path, 'real_images')
        generated_path = os.path.join(save_path,'fake_images')
        paths = [real_path, generated_path]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 计算 FID
        fid_value = fid_score.calculate_fid_given_paths(paths, batch_size=8, device=device, dims=2048, num_workers=0)
        # print("FID score:", fid_value)
        # 删除real_path
        # shutil.rmtree(real_path)
        # shutil.rmtree(generated_path)
        return fid_value

    def calculate_psnr(self,save_path):
        original_image_path = os.path.join(save_path, 'real_images')
        generated_image_path = os.path.join(save_path,'fake_images')
        original_images = sorted(os.listdir(original_image_path))
        generated_images = sorted(os.listdir(generated_image_path))

        total_psnr = 0.0
        num_images = len(original_images)

        for i in range(num_images):
            original = cv2.imread(os.path.join(original_image_path, original_images[i]))
            generated = cv2.imread(os.path.join(generated_image_path, generated_images[i]))

            mse = np.mean((original - generated) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                max_pixel_value = 255  # Assuming pixel values are in the range [0, 255]
                psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

            total_psnr += psnr

        average_psnr = total_psnr / num_images
        return average_psnr


    def run(self):
        ###### Definition of variables ######
        netG = self.create_models()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.Tensor
        input_A = Tensor(self.args.batchSize, self.args.channels,self.args.img_height, self.args.img_width)
        input_B = Tensor(self.args.batchSize, self.args.output_nc,self.args.img_height, self.args.img_width)
        # Dataset loader
        dataloader = self.load_dataset()
        ###################################

        ###### Testing######

        # Create output dirs if they don't exist
        save_path=self.create_save_path()
        print('-------保存路径：------------', save_path)

        for i, batch in enumerate(dataloader):
            print('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generate output
            fake_B = 0.5*(netG(real_A).data + 1.0) # 这里的fake_B是生成器生成的B图像，不是真实的B图像
            real_A = 0.5*(real_A.data + 1.0) # 这里的real_A是真实的A图像，不是生成器生成的A图像
            real_B = 0.5*(real_B.data + 1.0) # 这里的real_B是真实的B图像，不是生成器生成的B图像

            # 获取当前文件名
            file_name = os.path.basename(dataloader.dataset.files_A[i])

            if self.args.metric == True:
                os.makedirs(os.path.join(save_path,'real_images'), exist_ok=True)
                save_image(real_B, os.path.join(save_path,'real_images', file_name))
                os.makedirs(os.path.join(save_path,'fake_images'), exist_ok=True)
                save_image(fake_B, os.path.join(save_path,'fake_images', file_name))
            # 进行图像拼接
            output_img = torch.cat((real_B,real_A, fake_B), 3)
            # 保存图像
            save_image(output_img, os.path.join(save_path, file_name))



        if self.args.metric == True:
            # 计算FID
            fid_value = self.metric_func_FID(save_path)
            # 计算PSNR
            psnr_value = self.calculate_psnr(save_path)

            print("FID score:", fid_value)
            print("PSNR score:", psnr_value)

            # 保存FID和PSNR值
            with open(os.path.join(save_path, 'metic.txt'), 'a') as f:
                f.write('FID score: ' + str(fid_value))
                f.write('\n')
                f.write("PSNR score: " + str(psnr_value))

            real_path = os.path.join(save_path, 'real_images')
            generated_path = os.path.join(save_path,'fake_images')

            if args.save_imgs == False:
                # 删除real_path  generated_path
                shutil.rmtree(real_path)
                shutil.rmtree(generated_path)




if __name__ == '__main__':
    args = parser_args()
    detecter = Detecter(args)
    detecter.run()