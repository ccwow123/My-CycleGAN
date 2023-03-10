# -*- coding: utf-8 -*-
import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
import numpy as np
from torchvision.utils import save_image
'''这个模块设计了一个缓冲区，和学习率更新的函数
在更新discriminators的时候，用的是之前生成的图片，而不是最新的图片，所以设立图片缓冲区，可以存放50张之前生成的图片。
学习率初始为0.0003，总的epoch为50，在0-30的时候，学习率为0.0003，在30-50的时候，学习率逐渐线性减小为0，所以需要进行学习率的更新。
需要的变量有：总的训练epoch，当前的epoch，和开始进行衰减的epoch，即可实现lr的线性变化。
————————————————
版权声明：本文为CSDN博主「attacking tiger」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_39547794/article/details/125409710
'''
## 先前生成的样本的缓冲区
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):                       ## 放入一张图像，再从buffer里取一张出来
        to_return = []                                  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:          ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:          ## 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


## 设置学习率为初始学习率乘以给定lr_lambda函数的值
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):                                                ## (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"     ## 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):                                              ## return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
