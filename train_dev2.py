#!/usr/bin/python3

import argparse
import itertools
import os.path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.models import Generator
from utils.models import Discriminator
from utils.utils import ReplayBuffer
from utils.utils import LambdaLR
from utils.utils import Logger
from utils.utils import weights_init_normal
from utils.datasets import ImageDataset
from mytools import *
# python -m visdom.server
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="cycleGAN", type=str, help="选择模型")
    parser.add_argument('--dataroot', type=str, default=r'data\cap_b2cap_g - 副本', help='root directory of the dataset')
    parser.add_argument('--n_epochs', type=int, default=10, help='终止世代')
    parser.add_argument('--decay_epoch', type=int, default=1, help='开始线性衰减学习率为 0 的世代')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--size', type=int, default=32, help='数据裁剪的大小（假设为平方）')
    # 其他功能
    parser.add_argument('--pretrained', type=str, default='logs/03-14 15_58_39-cycleGAN', help='pretrained model path')
    parser.add_argument('--open-tensorboard', default=False, type=bool, help='使用tensorboard保存网络结构')
    # 默认
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--epoch', type=int, default=1, help='起始世代')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

    opt = parser.parse_args()
    print(opt)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


class Trainer(Train_base):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.args = args
        self.logger = None
        self.model_name = args.model_name
        self.log_dir = self._create_folder()
    # 创建模型
    def _create_model(self):
        # Networks
        netG_A2B = Generator(self.args.input_nc, self.args.output_nc)
        netG_B2A = Generator(self.args.output_nc, self.args.input_nc)
        netD_A = Discriminator(self.args.input_nc)
        netD_B = Discriminator(self.args.output_nc)

        if self.args.cuda:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()

        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

        # models = dict('netG_A2B'=netG_A2B, netG_B2A=netG_B2A, netD_A=netD_A, netD_B=netD_B)
        models = {'netG_A2B': netG_A2B, 'netG_B2A': netG_B2A, 'netD_A': netD_A, 'netD_B': netD_B}
        if self.args.pretrained:
            models = self._load_pretrained_model(models)

        return models
    # 加载预训练模型
    def _load_pretrained_model(self, models):
        pretrained_path = self.args.pretrained
        # 加载预训练权重
        models['netG_A2B'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_A2B.pth')))
        models['netG_B2A'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_B2A.pth')))
        models['netD_A'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_A.pth')))
        models['netD_B'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_B.pth')))
        print('加载预训练权重成功 {}'.format(pretrained_path))
        return models
    # 加载数据集
    def load_data(self):
        # Dataset loader
        transforms_ = [transforms.Resize(int(self.args.size * 1.12)),
                       transforms.CenterCrop(self.args.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader_ = DataLoader(ImageDataset(self.args.dataroot, transforms_=transforms_, unaligned=True),
                                 batch_size=self.args.batchSize, shuffle=True, num_workers=self.args.n_cpu)
        return dataloader_
    # 建立优化器和损失函数
    def create_optimizer(self, models):
        # Lossess
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()

        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(models['netG_A2B'].parameters(), models['netG_B2A'].parameters()),
            lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(models['netD_A'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(models['netD_B'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        # 加载预训练的优化器权重
        if self.args.pretrained:
            checkpoint = torch.load(os.path.join(self.args.pretrained, 'checkpoint.pth'))
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
            optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
            print('加载预训练优化器权重成功 {}'.format(self.args.pretrained))

        # 创建字典
        loss_dict = {'criterion_GAN': criterion_GAN, 'criterion_cycle': criterion_cycle,
                     'criterion_identity': criterion_identity}
        optimizer_dict = {'optimizer_G': optimizer_G, 'optimizer_D_A': optimizer_D_A, 'optimizer_D_B': optimizer_D_B}

        return loss_dict, optimizer_dict
    # 建立学习率调整策略
    def create_lr_scheduler(self,optimizer_dict):
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_dict['optimizer_G'],
                                                           lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                              self.args.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_dict['optimizer_D_A'],
                                                             lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                                self.args.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_dict['optimizer_D_B'],
                                                             lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                                self.args.decay_epoch).step)

        lr_dict = {'lr_scheduler_G': lr_scheduler_G, 'lr_scheduler_D_A': lr_scheduler_D_A,
                      'lr_scheduler_D_B': lr_scheduler_D_B}
        return lr_dict
    # 建立输入和标签内存
    def create_input_target(self):
        # Inputs & targets memory allocation
        ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
        Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.Tensor
        input_A = Tensor(self.args.batchSize, self.args.input_nc, self.args.size, self.args.size)
        input_B = Tensor(self.args.batchSize, self.args.output_nc, self.args.size, self.args.size)
        ## 全真，全假的标签
        target_real = Variable(Tensor(self.args.batchSize).fill_(1.0),
                               requires_grad=False)  ## 定义真实的图片label为1 ones((1, 1, 16, 16))
        target_fake = Variable(Tensor(self.args.batchSize).fill_(0.0),
                               requires_grad=False)  ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        self.input_dict = dict(A=input_A, B=input_B)
        self.target_dict = dict(real=target_real, fake=target_fake)
        self.buffer_dict = dict(A=fake_A_buffer, B=fake_B_buffer)
    # 训练一个epoch
    def train_one_epoch(self,model,train_loader,loss_dict,optimizer_dict,lr_dict):
        for i, batch in enumerate(train_loader):
            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
            real_A = Variable(self.input_dict['A'].copy_(batch['A']))# 真图像A
            real_B = Variable(self.input_dict['B'].copy_(batch['B']))# 真图像B
        ## -----------------
        ##  Train Generator
        ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        ## 反向传播更新的参数是生成网络里面的参数，
        ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
        ## -----------------
            ## -----------------
            model['netG_A2B'].train()
            model['netG_B2A'].train()

        ## Identity loss                                              ## A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
            loss_id_A = loss_dict['criterion_identity'](model['netG_B2A'](real_A), real_A)*5.0
            ## loss_id_A就是把图像A1放入 B2A 的生成器中，那当然生成图像A2的风格也得是A风格, 要让A1,A2的差距很小
            loss_id_B = loss_dict['criterion_identity'](model['netG_A2B'](real_B), real_B)*5.0

            loss_identity = (loss_id_A + loss_id_B) / 2  ## Identity loss

        ## GAN loss
            fake_B = model['netG_A2B'](real_A)  ## 用真图像A生成的假图像B
            pred_fake = model['netD_B'](fake_B)
            loss_GAN_AB = loss_dict['criterion_GAN'](pred_fake, self.target_dict['real'])  ## 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来
            fake_A = model['netG_B2A'](real_B)  ## 用真图像B生成的假图像A
            pred_fake = model['netD_A'](fake_A)
            loss_GAN_BA = loss_dict['criterion_GAN'](pred_fake, self.target_dict['real'])  ## 用A鉴别器鉴别假图像A，训练生成器的目的就是要让鉴别器以为假的是真的,假的太接近真的让鉴别器分辨不出来

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  ## GAN loss

        # Cycle loss 循环一致性损失
            recov_A = model['netG_B2A'](fake_B)  ## 之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
            loss_cycle_A = loss_dict['criterion_cycle'](recov_A, real_A)*10.0  ## realA和recovA的差距应该很小，以保证A,B间不仅风格有所变化，而且图片对应的的细节也可以保留
            recov_B = model['netG_A2B'](fake_A)
            loss_cycle_B = loss_dict['criterion_cycle'](recov_B, real_B)*10.0

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss                                                  ## 就是上面所有的损失都加起来
            loss_G = loss_identity + loss_GAN + loss_cycle
            optimizer_dict['optimizer_G'].zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_G.backward()  ## 将误差反向传播
            optimizer_dict['optimizer_G'].step()  ## 更新参数

            ## -----------------------
            ## Train Discriminator A
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            ## -----------------------
            ## 真的图像判别为真
            optimizer_dict['optimizer_D_A'].zero_grad()
            # Real loss
            pred_real = model['netD_A'](real_A)
            loss_D_real = loss_dict['criterion_GAN'](pred_real, self.target_dict['real'])
            # Fake loss
            fake_A = self.buffer_dict['A'].push_and_pop(model['netG_B2A'](real_B))
            pred_fake = model['netD_A'](fake_A.detach())
            loss_D_fake = loss_dict['criterion_GAN'](pred_fake, self.target_dict['fake'])

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_dict['optimizer_D_A'].step()
            ## -----------------------
            ## Train Discriminator B
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            ## -----------------------
            optimizer_dict['optimizer_D_B'].zero_grad()
            # 真的图像判别为真
            pred_real = model['netD_B'](real_B)
            loss_D_real = loss_dict['criterion_GAN'](pred_real, self.target_dict['real'])
            ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
            fake_B = self.buffer_dict['B'].push_and_pop(model['netG_A2B'](real_A))
            pred_fake = model['netD_B'](fake_B.detach())
            loss_D_fake = loss_dict['criterion_GAN'](pred_fake, self.target_dict['fake'])

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_dict['optimizer_D_B'].step()

        # DA and DB loss
            loss_D = (loss_D_A + loss_D_B) / 2
        # Progress report (http://localhost:8097)
            self.logger.log({'loss_G': loss_G, 'loss_G_identity': loss_identity, 'loss_G_GAN': loss_GAN,
                        'loss_G_cycle': loss_cycle, 'loss_D': loss_D},
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
            log_loss = {'loss_G': loss_G, 'loss_G_identity': loss_identity, 'loss_G_GAN': loss_GAN,
                    'loss_G_cycle': loss_cycle, 'loss_D': loss_D}
        return log_loss
    # 保存训练过程中的信息
    def save_logs(self,i,log_loss, lr_dict, models):
        # 将损失写入tensorboard
        self.tb.add_scalar('loss_G', log_loss['loss_G'], i)
        self.tb.add_scalar('loss_G_identity', log_loss['loss_G_identity'], i)
        self.tb.add_scalar('loss_G_GAN', log_loss['loss_G_GAN'], i)
        self.tb.add_scalar('loss_G_cycle', log_loss['loss_G_cycle'], i)
        self.tb.add_scalar('loss_D', log_loss['loss_D'], i)
        # 将学习率写入tensorboard
        self.tb.add_scalar('lr_G', lr_dict['lr_scheduler_G'].get_last_lr()[0], i)
        self.tb.add_scalar('lr_D_A', lr_dict['lr_scheduler_D_A'].get_last_lr()[0], i)
        self.tb.add_scalar('lr_D_B', lr_dict['lr_scheduler_D_B'].get_last_lr()[0], i)
        # 将模型写入tensorboard
        if self.args.open_tensorboard is True:
            self.tb.add_graph(models['netG_A2B'], self.input_dict['input_A'])
            # self.tb.add_graph(models['netG_B2A'], self.input_dict['input_B'])
            # self.tb.add_graph(models['netD_A'], self.input_dict['input_A'])
            # self.tb.add_graph(models['netD_B'], self.input_dict['input_B'])

    def run(self):
        # 加载检查点
        if self.args.pretrained :
            checkpoint = torch.load(os.path.join(self.args.pretrained, 'checkpoint.pth'))
            self.args.epoch = checkpoint['epoch']
            self.args.n_epochs += self.args.epoch
            self.args.decay_epoch += self.args.epoch
            print('已经训练： %d epoch' % self.args.epoch)
        # 输入和目标内存分配
        self.create_input_target()
        # 数据加载
        train_loader = self.load_data()
        # 模型创建
        models = self._create_model()
        # 损失函数 优化器创建
        loss_dict, optimizer_dict = self.create_optimizer(models)
        # 学习率调整
        lr_dict= self.create_lr_scheduler(optimizer_dict)
        # Visdom
        self.logger = Logger(self.args.epoch,self.args.n_epochs, len(train_loader))
        # 模型训练
        for epoch in range(self.args.epoch, self.args.n_epochs+1):
            # 训练
            log_loss = self.train_one_epoch(models,train_loader,loss_dict,optimizer_dict,lr_dict)
            # Update learning rates
            lr_dict['lr_scheduler_G'].step()
            lr_dict['lr_scheduler_D_A'].step()
            lr_dict['lr_scheduler_D_B'].step()
            # 保存日志
            self.save_logs(epoch, log_loss, lr_dict, models)

            # 保存模型
            # if epoch % self.args.save_epoch == 0 and epoch != self.args.n_epochs - 1:
            torch.save(models['netG_A2B'].state_dict(), f"{self.log_dir}/netG_A2B.pth")
            torch.save(models['netG_B2A'].state_dict(), f"{self.log_dir}/netG_B2A.pth")
            torch.save(models['netD_A'].state_dict(), f"{self.log_dir}/netD_A.pth")
            torch.save(models['netD_B'].state_dict(), f"{self.log_dir}/netD_B.pth")
            # 保存优化器及学习率
            torch.save({
                'epoch': self.args.n_epochs,
                'optimizer_G': optimizer_dict['optimizer_G'].state_dict(),
                'optimizer_D_A': optimizer_dict['optimizer_D_A'].state_dict(),
                'optimizer_D_B': optimizer_dict['optimizer_D_B'].state_dict(),
            }, f"{self.log_dir}/checkpoint.pth")



if __name__ == '__main__':
    # 参数解析
    args = parse_args()
    # 创建模型
    model = Trainer(args)
    # 模型训练
    model.run()
