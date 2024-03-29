
import argparse
import csv
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils import Generator,Discriminator,weights_init_normal
from utils.datasets import ImageDataset_pix2pix
from utils.models_pix2pix import GeneratorUNet as Generator_pix2pix , Discriminator as Discriminator_pix2pix 
from utils.models_pix2pix import GeneratorUNet_A,GeneratorUNet_A_en
from utils.models_pix2pix import Discriminator2, Discriminator_SN
from utils.models_pix2pix import SPADEGenerator,NL_Generator
import torch.nn.functional as F
import torch


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")

    parser.add_argument("--dataset", type=str, default=r"E:\datasets\_using\pix_MLCCn6", help="name of the dataset")

    parser.add_argument("--A2B", default=True, help="翻译方向")
    parser.add_argument("--Discriminator", type=str, default="ori",choices=["ori",'2','SN'] ,help="判别器类型")
    parser.add_argument("--Generator", type=str, default="NL",choices=["ori",'A','A_en','SPADE'] , help="生成器类型")
    parser.add_argument("--wgangp",  default=True, help="是否使用WGAN-GP")


    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=250, help="多少epoch进行一次模型保存")
    # 一般不变
    parser.add_argument("--sample_interval", type=int, default=500, help="从发生器对图像进行采样之间的间隔")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")

    opt = parser.parse_args()
    print(opt)
    return opt


def create_log(args):
    # 用来保存训练以及验证过程中信息的文件夹
    if not os.path.exists("logs_pix"):
        os.mkdir("logs_pix")
    # 创建时间+模型名文件夹
    time_str = datetime.datetime.now().strftime("%m-%d %H_%M_%S-")
    log_dir = os.path.join("logs_pix", args.Discriminator + "-" + args.Generator + "-" + args.dataset_name)
    if args.wgangp:
        log_dir = log_dir + "-WGAN_GP"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    results_file = os.path.join(log_dir, time_str + "results.txt")
    # 实例化tensborad
    tb = SummaryWriter(log_dir=log_dir)
    # 实例化wandb
    # config = {'data-path': self.args.data_path, 'batch-size': self.args.batch_size}
    # self.wandb = wandb.init(project='newproject',name='每次改一下名称', config=config, dir=log_dir)
    return log_dir, results_file, tb

def craete_model(opt):
    # 生成器
    if opt.Generator == "ori":
        generator = Generator_pix2pix(opt.channels, opt.channels)
    elif opt.Generator == "A":
        generator = GeneratorUNet_A(opt.channels, opt.channels)
    elif opt.Generator == "A_en":
        generator = GeneratorUNet_A_en(opt.channels, opt.channels)
    elif opt.Generator == "SPADE":
        generator = SPADEGenerator(opt.channels, opt.channels)
    elif opt.Generator == "NL":
        generator = NL_Generator(opt.channels, opt.channels)
    else:
        raise Exception("Generator type not implemented!")

    # 判别器
    if opt.Discriminator == "ori":
        discriminator = Discriminator_pix2pix(opt.channels)
    elif opt.Discriminator == "2":
        discriminator = Discriminator2(opt.channels)
    elif opt.Discriminator == "SN":
        discriminator = Discriminator_SN(opt.channels)
    else:
        raise Exception("Discriminator type not implemented!")
    return generator,discriminator

def initialize_weights(model):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
    return model

# Wgan-gp
def compute_gradient_penalty(discriminator, real_A, real_B, fake_B):
    """计算梯度惩罚"""
    batch_size = real_A.size(0)

    # 在真实图像和生成图像之间进行插值采样
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_A.device)
    interpolated_A = alpha * real_A + (1 - alpha) * fake_B
    interpolated_A.requires_grad_(True)

    # 计算插值样本通过判别器的输出
    d_interpolated = discriminator(interpolated_A, fake_B)

    # 计算梯度惩罚项
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_A,
                                    grad_outputs=torch.ones(d_interpolated.size()).to(real_A.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
def main(opt):
    # 设备为GPU
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")
    # 初始化日志文件夹
    opt.dataset_name = os.path.basename(opt.dataset) # 获取数据集名称
    log_dir, results_file, tb = create_log(opt) # 创建日志文件夹,返回配置文件以及tensorboard实例
    checkpoint_path = os.path.join(log_dir, "saved_models") # 保存模型的文件夹
    os.makedirs(checkpoint_path, exist_ok=True)
    imgs_save_path = os.path.join(log_dir, "images") # 保存图片的文件夹
    os.makedirs(imgs_save_path, exist_ok=True)


    # 损失函数创建
    criterion_GAN = torch.nn.MSELoss() # GAN损失  原版
    criterion_pixelwise = torch.nn.L1Loss() # 像素损失

    # L1像素损失在平移图像和真实图像之间的损失
    lambda_pixel = 100

    # 计算图像鉴别器（PatchGAN）的输出大小
    # patch = (1, opt.img_height // 2 ** 5, opt.img_width // 2 ** 5) #原版
    # patch = (1, 8, 8)

    # 初始化生成器和判别器
    generator,discriminator = craete_model(opt)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(os.path.join(checkpoint_path, "generator_%d.pth" % (opt.epoch))))
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_path, "discriminator_%d.pth" % (opt.epoch))))
    else:
        # Initialize weights
        generator=initialize_weights(generator)
        discriminator=initialize_weights(discriminator)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 配置数据加载器
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset_pix2pix( opt.dataset, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset_pix2pix( opt.dataset, transforms_=transforms_, mode="test"),
        batch_size=10,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_images(batches_done,A2B):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        if A2B == True:
            real_A = Variable(imgs["A"].type(Tensor))
            real_B = Variable(imgs["B"].type(Tensor))
        else:
            real_A = Variable(imgs["B"].type(Tensor))
            real_B = Variable(imgs["A"].type(Tensor))
        # real_A = Variable(imgs["A"].type(Tensor))
        # real_B = Variable(imgs["B"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, os.path.join(imgs_save_path, "%s.png" % (batches_done)), nrow=5, normalize=True)


    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    if opt.wgangp != True:
        # 原版 ori 训练
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):

                # Model inputs
                if opt.A2B== True:
                    real_A = Variable(batch["A"].type(Tensor))
                    real_B = Variable(batch["B"].type(Tensor))
                else:
                    real_A = Variable(batch["B"].type(Tensor))
                    real_B = Variable(batch["A"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *discriminator.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *discriminator.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # GAN loss
                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                loss_real = criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)

                # Total loss
                loss_G = loss_real + lambda_pixel * loss_pixel

                loss_G.backward()

                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_B, real_A)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                # 生成样例
                if batches_done % opt.sample_interval == 0:
                    sample_images(batches_done,opt.A2B)
            #  打印log
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s'
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_real.item(),
                    time_left,
                )
            )

            # 记录日志
            tb.add_scalar('D/loss_D', loss_D.item(), epoch)
            tb.add_scalar('G/loss_G', loss_G.item(), epoch)
            tb.add_scalar('G/loss_pixel', loss_pixel.item(), epoch)
            tb.add_scalar('G/loss_adv', loss_real.item(), epoch)
            # 创建csv文件
            header_list = ["epoch", 'loss_D','loss_G','loss_pixel','loss_adv']
            with open(log_dir + '/val_log.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header_list)
                if epoch == 0:
                    writer.writeheader()
                writer.writerow({'epoch': epoch, 'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'loss_pixel': loss_pixel.item(), 'loss_adv': loss_real.item()})
            # 保存args参数
            with open(log_dir + '/results_file.txt', 'w') as f:
                f.write(str(opt))

            # 保存模型
            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            if (epoch+1) % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), os.path.join(checkpoint_path, "generator_%d.pth" % (epoch+1)))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_path, "discriminator_%d.pth" % (epoch+1)))

    else:  # wgan-gp 训练
        lambda_gp = 100  # 梯度惩罚系数,越大越好
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):
                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *discriminator.output_shape))),
                                 requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *discriminator.output_shape))),
                                requires_grad=False)

                # ---------------------
                # 更新判别器的参数
                # ---------------------
                optimizer_D.zero_grad()
                discriminator.zero_grad()

                fake_B = generator(real_A) # 生成器生成的假图像

                # Real loss
                pred_real = discriminator(real_B, real_A) # 判别器判别真图像,真图像输出为1
                loss_real = criterion_GAN(pred_real, valid) # GAN loss

                # 计算 Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)

                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A) # 判别器判别假图像,假图像输出为0
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake) # 判别器的总loss
                gradient_penalty = compute_gradient_penalty(discriminator, real_A, real_B, fake_B) # 计算梯度惩罚
                loss_D_total = lambda_gp * gradient_penalty + loss_D + loss_pixel # 判别器的总loss

                loss_D_total.backward()
                optimizer_D.step()

                # ---------------------
                # 更新生成器的参数
                # ---------------------
                optimizer_G.zero_grad()
                generator.zero_grad()

                # GAN loss
                fake_B = generator(real_A) # 生成器生成的假图像
                pred_fake = discriminator(real_A, fake_B) # 判别器判别假图像,假图像输出为1
                loss_real = criterion_GAN(pred_fake, valid)

                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)

                # Total loss
                loss_G = loss_real + lambda_pixel * loss_pixel

                loss_G.backward()
                optimizer_G.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # 生成样例
                if batches_done % opt.sample_interval == 0:
                    sample_images(batches_done, opt.A2B)

            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s'
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_total.item(),
                    loss_G.item(),
                    time_left,
                ))

            # 记录日志
            tb.add_scalar('D/loss_D', loss_D.item(), epoch)
            tb.add_scalar('G/loss_G', loss_G.item(), epoch)
            tb.add_scalar('G/loss_pixel', loss_pixel.item(), epoch)
            tb.add_scalar('G/loss_adv', loss_real.item(), epoch)
            # 创建csv文件
            header_list = ["epoch", 'loss_D','loss_G','loss_pixel','loss_adv']
            with open(log_dir + '/val_log.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header_list)
                if epoch == 0:
                    writer.writeheader()
                writer.writerow({'epoch': epoch, 'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'loss_pixel': loss_pixel.item(), 'loss_adv': loss_real.item()})
            # 保存args参数
            with open(log_dir + '/results_file.txt', 'w') as f:
                f.write(str(opt))

            # 保存模型
            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            if (epoch+1) % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), os.path.join(checkpoint_path, "generator_%d.pth" % (epoch+1)))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_path, "discriminator_%d.pth" % (epoch+1)))

if __name__ == '__main__':
    opt = parser_args()
    main(opt)