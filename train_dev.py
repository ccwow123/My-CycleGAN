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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="cycleGAN", type=str, help="选择模型")
    parser.add_argument('--n_epochs', type=int, default=10, help='终止世代')
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default=r'data\cap_b2cap_g', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='开始线性衰减学习率为 0 的世代')
    parser.add_argument('--size', type=int, default=256, help='数据裁剪的大小（假设为平方）')
    # 其他功能
    parser.add_argument('--pretrained', type=str, default='output_ori', help='pretrained model path')
    parser.add_argument('--open-tensorboard', default=False, type=bool, help='使用tensorboard保存网络结构')
    # 默认
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--epoch', type=int, default=0, help='起始世代')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

    opt = parser.parse_args()
    print(opt)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


class Trainer:
    def __init__(self, args):
        self.args = args
        self.logger = None

    # 创建文件夹
    def create_folder(self):
        # 用来保存训练以及验证过程中信息
        if not os.path.exists("logs"):
            os.mkdir("logs")
        # 创建时间+模型名文件夹
        time_str = datetime.datetime.now().strftime("%m-%d %H_%M_%S-")
        log_dir = os.path.join("logs", time_str + self.args.model_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.results_file = log_dir + "/{}_results{}.txt".format(self.args.model_name, time_str)
        # 实例化tensborad
        self.tb = SummaryWriter(log_dir=log_dir)
        return log_dir

    # 数据集加载
    def load_dataset(self):
        # Dataset loader
        transforms_ = [transforms.Resize(int(self.args.size * 1.12)),
                       transforms.CenterCrop(self.args.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader_ = DataLoader(ImageDataset(self.args.dataroot, transforms_=transforms_, unaligned=True),
                                 batch_size=self.args.batchSize, shuffle=True, num_workers=self.args.n_cpu)
        return dataloader_

    # 模型创建
    def create_model(self):
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
            models = self.load_pretrained_model(models)

        return models

    # 加载预训练模型
    def load_pretrained_model(self, models):
        pretrained_path = self.args.pretrained
        # 加载预训练权重
        models['netG_A2B'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_A2B.pth')))
        models['netG_B2A'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_B2A.pth')))
        models['netD_A'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_A.pth')))
        models['netD_B'].load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_B.pth')))
        print('加载预训练权重成功 {}'.format(pretrained_path))
        return models

    # 优化器创建
    def create_optimizer(self, models):
        # Lossess
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        optimizer_G = torch.optim.Adam(
            itertools.chain(models['netG_A2B'].parameters(), models['netG_B2A'].parameters()),
            lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(models['netD_A'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(models['netD_B'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                           lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                              self.args.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                             lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                                self.args.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                             lr_lambda=LambdaLR(self.args.n_epochs, self.args.epoch,
                                                                                self.args.decay_epoch).step)
        # 创建字典
        self.loss_dict = dict(criterion_GAN=criterion_GAN, criterion_cycle=criterion_cycle,
                              criterion_identity=criterion_identity)
        self.optimizer_dict = dict(optimizer_G=optimizer_G, optimizer_D_A=optimizer_D_A, optimizer_D_B=optimizer_D_B)
        lr_scheduler_dict = dict(lr_scheduler_G=lr_scheduler_G, lr_scheduler_D_A=lr_scheduler_D_A,
                                 lr_scheduler_D_B=lr_scheduler_D_B)
        return lr_scheduler_dict

    # 输入和目标内存分配
    def create_input_target(self):
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.Tensor
        input_A = Tensor(self.args.batchSize, self.args.input_nc, self.args.size, self.args.size)
        input_B = Tensor(self.args.batchSize, self.args.output_nc, self.args.size, self.args.size)
        target_real = Variable(Tensor(self.args.batchSize).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.args.batchSize).fill_(0.0), requires_grad=False)

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        self.input_dict = dict(input_A=input_A, input_B=input_B)
        self.target_dict = dict(target_real=target_real, target_fake=target_fake)
        self.buffer_dict = dict(fake_A_buffer=fake_A_buffer, fake_B_buffer=fake_B_buffer)

    def _generators(self, real_dict, models):
        self.optimizer_dict['optimizer_G'].zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = models['netG_A2B'](self.input_dict['input_B'])
        loss_identity_B = self.loss_dict['criterion_identity'](same_B, self.input_dict['input_B']) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = models['netG_B2A'](self.input_dict['input_A'])
        loss_identity_A = self.loss_dict['criterion_identity'](same_A, self.input_dict['input_A']) * 5.0

        # GAN loss
        fake_B = models['netG_A2B'](self.input_dict['input_A'])
        pred_fake = models['netD_B'](fake_B)
        loss_GAN_A2B = self.loss_dict['criterion_GAN'](pred_fake, self.target_dict['target_real'])

        fake_A = models['netG_B2A'](self.input_dict['input_B'])
        pred_fake = models['netD_A'](fake_A)
        loss_GAN_B2A = self.loss_dict['criterion_GAN'](pred_fake, self.target_dict['target_real'])

        # Cycle loss
        recovered_A = models['netG_B2A'](fake_B)
        loss_cycle_ABA = self.loss_dict['criterion_cycle'](recovered_A, real_dict['real_A']) * 10.0

        recovered_B = models['netG_A2B'](fake_A)
        loss_cycle_BAB = self.loss_dict['criterion_cycle'](recovered_B, real_dict['real_B']) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        self.optimizer_dict['optimizer_G'].step()
        ###################################
        loss_G_dict = dict(loss_G=loss_G, loss_identity_A=loss_identity_A, loss_identity_B=loss_identity_B,
                           loss_GAN_A2B=loss_GAN_A2B, loss_GAN_B2A=loss_GAN_B2A, loss_cycle_ABA=loss_cycle_ABA,
                           loss_cycle_BAB=loss_cycle_BAB)
        return loss_G_dict

    def _discriminator_A(self, real_dict, models):
        self.optimizer_dict['optimizer_D_A'].zero_grad()
        # Real loss
        pred_real = models['netD_A'](real_dict['real_A'])
        loss_D_real = self.loss_dict['criterion_GAN'](pred_real, self.target_dict['target_real'])
        # Fake loss
        fake_A = self.buffer_dict['fake_A_buffer'].push_and_pop(models['netG_B2A'](real_dict['real_B']))
        pred_fake = models['netD_A'](fake_A.detach())
        loss_D_fake = self.loss_dict['criterion_GAN'](pred_fake, self.target_dict['target_fake'])

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        self.optimizer_dict['optimizer_D_A'].step()
        ###################################
        return loss_D_A, fake_A

    def _discriminator_B(self, real_dict, models):
        self.optimizer_dict['optimizer_D_B'].zero_grad()
        # Real loss
        pred_real = models['netD_B'](real_dict['real_B'])
        loss_D_real = self.loss_dict['criterion_GAN'](pred_real, self.target_dict['target_real'])
        # Fake loss
        fake_B = self.buffer_dict['fake_B_buffer'].push_and_pop(models['netG_A2B'](real_dict['real_A']))
        pred_fake = models['netD_B'](fake_B.detach())
        loss_D_fake = self.loss_dict['criterion_GAN'](pred_fake, self.target_dict['target_fake'])

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        self.optimizer_dict['optimizer_D_B'].step()
        ###################################
        return loss_D_B, fake_B

    # 一轮训练
    def train_one_epoch(self, dataloader, models, logger):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(self.input_dict['input_A'].copy_(batch['A']))
            real_B = Variable(self.input_dict['input_B'].copy_(batch['B']))
            real_dict = dict(real_A=real_A, real_B=real_B)

            ###### Generators A2B and B2A ######
            loss_G_dict = self._generators(real_dict, models)
            ###### Discriminator A ######
            loss_D_A, fake_A = self._discriminator_A(real_dict, models)
            ###### Discriminator B ######
            loss_D_B, fake_B = self._discriminator_B(real_dict, models)
            # 打包损失
            loss_dict = dict(loss_G_dict=loss_G_dict, loss_D_A=loss_D_A, loss_D_B=loss_D_B)

            # 记录损失
            loss_G = loss_dict['loss_G_dict']['loss_G']
            loss_G_identity = loss_dict['loss_G_dict']['loss_identity_A'] + loss_dict['loss_G_dict']['loss_identity_B']
            loss_G_GAN = loss_dict['loss_G_dict']['loss_GAN_A2B'] + loss_dict['loss_G_dict']['loss_GAN_B2A']
            loss_G_cycle = loss_dict['loss_G_dict']['loss_cycle_ABA'] + loss_dict['loss_G_dict']['loss_cycle_BAB']
            loss_D = loss_dict['loss_D_A'] + loss_dict['loss_D_B']
            # Progress report (http://localhost:8097)
            logger.log(losses={'loss_G': loss_G, 'loss_G_identity': loss_G_identity, 'loss_G_GAN': loss_G_GAN,
                               'loss_G_cycle': loss_G_cycle, 'loss_D': loss_D},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        return loss_dict

    # 写入tensorboard
    def _write_tb(self, epoch, loss_dict, lr_scheduler_dict, models):
        # 将损失写入tensorboard
        self.tb.add_scalar('loss_G', loss_dict['loss_G_dict']['loss_G'], epoch)
        self.tb.add_scalar('loss_G_identity', loss_dict['loss_G_dict']['loss_identity_A'] + loss_dict['loss_G_dict']['loss_identity_B'], epoch)
        self.tb.add_scalar('loss_G_GAN', loss_dict['loss_G_dict']['loss_GAN_A2B'] + loss_dict['loss_G_dict']['loss_GAN_B2A'], epoch)
        self.tb.add_scalar('loss_G_cycle', loss_dict['loss_G_dict']['loss_cycle_ABA'] + loss_dict['loss_G_dict']['loss_cycle_BAB'], epoch)
        self.tb.add_scalar('loss_D_A', loss_dict['loss_D_A'] , epoch)
        self.tb.add_scalar('loss_D_B', loss_dict['loss_D_B'] , epoch)
        # 将学习率写入tensorboard
        self.tb.add_scalar('lr_G', lr_scheduler_dict['lr_scheduler_G'].get_last_lr()[0], epoch)
        self.tb.add_scalar('lr_D_A', lr_scheduler_dict['lr_scheduler_D_A'].get_last_lr()[0], epoch)
        self.tb.add_scalar('lr_D_B', lr_scheduler_dict['lr_scheduler_D_B'].get_last_lr()[0], epoch)

        if self.args.open_tensorboard is True:
            self.tb.add_graph(models['netG_A2B'], self.input_dict['input_A'])
            self.tb.add_graph(models['netG_B2A'], self.input_dict['input_B'])
            self.tb.add_graph(models['netD_A'], self.input_dict['input_A'])
            self.tb.add_graph(models['netD_B'], self.input_dict['input_B'])


    # 模型训练
    def run(self):
        log_dir = self.create_folder()
        # 数据加载
        dataloader = self.load_dataset()
        # 模型创建
        models = self.create_model()
        # 优化器创建
        lr_scheduler_dict = self.create_optimizer(models)
        # 输入和目标内存分配
        self.create_input_target()
        # Loss plot
        logger = Logger(self.args.n_epochs, len(dataloader))
        # 模型训练
        best_loss=[2]*4
        for epoch in range(self.args.epoch, self.args.n_epochs):
            # 训练
            loss_dict = self.train_one_epoch(dataloader, models, logger)
            # Update learning rates
            lr_scheduler_dict['lr_scheduler_G'].step()
            lr_scheduler_dict['lr_scheduler_D_A'].step()
            lr_scheduler_dict['lr_scheduler_D_B'].step()
            # 保存日志
            self._write_tb(epoch, loss_dict, lr_scheduler_dict, models)

            # 保存模型
            # if epoch % self.args.save_epoch == 0 and epoch != self.args.n_epochs - 1:
            torch.save(models['netG_A2B'].state_dict(), f"{log_dir}/netG_A2B.pth")
            torch.save(models['netG_B2A'].state_dict(), f"{log_dir}/netG_B2A.pth")
            torch.save(models['netD_A'].state_dict(), f"{log_dir}/netD_A.pth")
            torch.save(models['netD_B'].state_dict(), f"{log_dir}/netD_B.pth")
            # elif loss_dict['loss_G_dict']['loss_GAN_A2B'] < best_loss[0]:
            #     best_loss[0] = loss_dict['loss_G_dict']['loss_GAN_A2B']
            #     torch.save(models['netG_A2B'].state_dict(), f"{log_dir}/netG_A2B.pth")
            # elif loss_dict['loss_G_dict']['loss_GAN_B2A'] < best_loss[1]:
            #     best_loss[1] = loss_dict['loss_G_dict']['loss_GAN_B2A']
            #     torch.save(models['netG_B2A'].state_dict(), f"{log_dir}/netG_B2A.pth")
            # elif loss_dict['loss_D_A'] < best_loss[2]:
            #     best_loss[2] = loss_dict['loss_G_dict']['loss_D_A']
            #     torch.save(models['netD_A'].state_dict(), f"{log_dir}/netD_A.pth")
            # elif loss_dict['loss_D_B'] < best_loss[3]:
            #     best_loss[3] = loss_dict['loss_G_dict']['loss_D_B']
            #     torch.save(models['netD_B'].state_dict(), f"{log_dir}/netD_B.pth")


if __name__ == '__main__':
    # 参数解析
    args = parse_args()
    # 创建模型
    model = Trainer(args)
    # 模型训练
    model.run()
