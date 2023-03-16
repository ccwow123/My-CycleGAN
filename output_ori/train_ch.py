import argparse
import os
from tkinter import Image
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from utils_ch.models import *
from utils_ch.datasets  import *
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="data/E_skew2Good",help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
# opt = parser.parse_args(args=[])                 ## 在colab中运行时，换为此行
print(opt)

## 创建文件夹
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("save/%s" % opt.dataset_name, exist_ok=True)

## input_shape:(3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

## 创建生成器，判别器对象
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

## 损失函数
## MES 二分类的交叉熵
## L1loss 相比于L2 Loss保边缘
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

## 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

## 如果epoch == 0，初始化模型参数; 如果epoch == n, 载入训练到第n轮的预训练模型
if opt.epoch != 0:
    # 载入训练到第n轮的预训练模型
    G_AB.load_state_dict(torch.load("saved/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 初始化模型参数
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

## 定义优化函数,优化函数的学习率为0.0003
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

## 学习率更行进程
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

## 先前生成的样本的缓冲区
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

## 图像 transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12)),  ## 图片放大1.12倍
    transforms.RandomCrop((opt.img_height, opt.img_width)),  ## 随机裁剪成原来的大小
    transforms.RandomHorizontalFlip(),  ## 随机水平翻转
    transforms.ToTensor(),  ## 变为Tensor数据
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  ## 正则化
]

## Training data loader
dataloader = DataLoader(  ## 改成自己存放文件的目录
    ImageDataset("data/cap_b2cap_g", transforms_=transforms_, unaligned=True),
    ## "./datasets/facades" , unaligned:设置非对其数据
    batch_size=opt.batch_size,  ## batch_size = 1
    shuffle=True,
    num_workers=opt.n_cpu,
)
## Test data loader
val_dataloader = DataLoader(
    ImageDataset("data/cap_b2cap_g", transforms_=transforms_, unaligned=True, mode="test"),  ## "./datasets/facades"
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


## 每间隔100次打印图片
def sample_images(batches_done):  ## （100/200/300/400...）
    """保存测试集中生成的样本"""
    imgs = next(iter(val_dataloader))  ## 取一张图像
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"]).cuda()  ## 取一张真A
    fake_B = G_AB(real_A)  ## 用真A生成假B
    real_B = Variable(imgs["B"]).cuda()  ## 去一张真B
    fake_A = G_BA(real_B)  ## 用真B生成假A
    # Arange images along x-axis
    ## make_grid():用于把几个图像按照网格排列的方式绘制出来
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    ## 把以上图像都拼接起来，保存为一张大图片
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


def train():
    # ----------
    #  Training
    # ----------
    prev_time = time.time()  ## 开始时间
    for epoch in range(opt.epoch, opt.n_epochs):  ## for epoch in (0, 50)
        for i, batch in enumerate(
                dataloader):  ## batch is a dict, batch['A']:(1, 3, 256, 256), batch['B']:(1, 3, 256, 256)
            #       print('here is %d' % i)
            ## 读取数据集中的真图片
            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
            real_A = Variable(batch["A"]).cuda()  ## 真图像A
            real_B = Variable(batch["B"]).cuda()  ## 真图像B

            ## 全真，全假的标签
            valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape)),
                             requires_grad=False).cuda()  ## 定义真实的图片label为1 ones((1, 1, 16, 16))
            fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape)),
                            requires_grad=False).cuda()  ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

            ## -----------------
            ##  Train Generator
            ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            ## 反向传播更新的参数是生成网络里面的参数，
            ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
            ## -----------------
            G_AB.train()
            G_BA.train()

            ## Identity loss                                              ## A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
            loss_id_A = criterion_identity(G_BA(real_A),
                                           real_A)  ## loss_id_A就是把图像A1放入 B2A 的生成器中，那当然生成图像A2的风格也得是A风格, 要让A1,A2的差距很小
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2  ## Identity loss

            ## GAN loss
            fake_B = G_AB(real_A)  ## 用真图像A生成的假图像B
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)  ## 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来
            fake_A = G_BA(real_B)  ## 用真图像B生成的假图像A
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)  ## 用A鉴别器鉴别假图像A，训练生成器的目的就是要让鉴别器以为假的是真的,假的太接近真的让鉴别器分辨不出来

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  ## GAN loss

            # Cycle loss 循环一致性损失
            recov_A = G_BA(fake_B)  ## 之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
            loss_cycle_A = criterion_cycle(recov_A, real_A)  ## realA和recovA的差距应该很小，以保证A,B间不仅风格有所变化，而且图片对应的的细节也可以保留
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss                                                  ## 就是上面所有的损失都加起来
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            optimizer_G.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_G.backward()  ## 将误差反向传播
            optimizer_G.step()  ## 更新参数

            ## -----------------------
            ## Train Discriminator A
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            ## -----------------------
            ## 真的图像判别为真
            loss_real = criterion_GAN(D_A(real_A), valid)
            ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            optimizer_D_A.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_D_A.backward()  ## 将误差反向传播
            optimizer_D_A.step()  ## 更新参数

            ## -----------------------
            ## Train Discriminator B
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            ## -----------------------
            # 真的图像判别为真
            loss_real = criterion_GAN(D_B(real_B), valid)
            ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            optimizer_D_B.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_D_B.backward()  ## 将误差反向传播
            optimizer_D_B.step()  ## 更新参数

            loss_D = (loss_D_A + loss_D_B) / 2

            ## ----------------------
            ##  打印日志Log Progress
            ## ----------------------

            ## 确定剩下的大约时间  假设当前 epoch = 5， i = 100
            batches_done = epoch * len(dataloader) + i  ## 已经训练了多长时间 5 * 400 + 100 次
            batches_left = opt.n_epochs * len(dataloader) - batches_done  ## 还剩下 50 * 400 - 2100 次
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))  ## 还需要的时间 time_left = 剩下的次数 * 每次的时间
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # 每训练100张就保存一组测试集中的图片
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    ## 训练结束后，保存模型
    torch.save(G_AB.state_dict(), "save/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    torch.save(G_BA.state_dict(), "save/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_A.state_dict(), "save/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_B.state_dict(), "save/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
    print("save my model finished !!")
    #    ## 每间隔几个epoch保存一次模型
    #     if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    #         # Save model checkpoints
    #         torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))


## 函数的起始
if __name__ == '__main__':
    train()
