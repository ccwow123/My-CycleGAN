# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

## 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.


##############################
##  残差块儿ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(                     ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),                      ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),     ## 卷积
            nn.InstanceNorm2d(in_features),             ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),                      ## 非线性激活
            nn.ReflectionPad2d(1),                      ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),     ## 卷积
            nn.InstanceNorm2d(in_features),             ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
        )

    def forward(self, x):                               ## 输入为 一张图像
        return x + self.block(x)                        ## 输出为 图像加上网络的残差输出



##############################
##  生成器网络GeneratorResNet
##############################
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):   ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]                           ## 输入通道数channels = 3

        ## 初始化网络结构
        out_features = 64                                   ## 输出特征数out_features = 64
        model = [                                           ## model = [Pad + Conv + Norm + ReLU]
            nn.ReflectionPad2d(channels),                   ## ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.Conv2d(channels, out_features, 7),           ## Conv2d(3, 64, 7)
            nn.InstanceNorm2d(out_features),                ## InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),                          ## 非线性激活
        ]
        in_features = out_features                          ## in_features = 64

        ## 下采样，循环2次
        for _ in range(2):
            out_features *= 2                                                   ## out_features = 128 -> 256
            model += [                                                          ## (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features                                          ## in_features = 256

        # 残差块儿，循环9次
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]                              ## model += [pad + conv + norm + relu + pad + conv + norm]

        # 上采样两次
        for _ in range(2):
            out_features //= 2                                                  ## out_features = 128 -> 64
            model += [                                                          ## model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features                                          ## out_features = 64

        ## 网络输出层                                                            ## model += [pad + conv + tanh]
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]    ## 将(3)的数据每一个都映射到[-1, 1]之间

        self.model = nn.Sequential(*model)

    def forward(self, x):           ## 输入(1, 3, 256, 256)
        return self.model(x)        ## 输出(1, 3, 256, 256)



##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape                                       ## input_shape:(3， 256， 256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)                  ## output_shape = (1, 16, 16)

        def discriminator_block(in_filters, out_filters, normalize=True):           ## 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]   ## layer += [conv + norm + relu]
            if normalize:                                                           ## 每次卷积尺寸会缩小一半，共卷积了4次
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),        ## layer += [conv(3, 64) + relu]
            *discriminator_block(64, 128),                              ## layer += [conv(64, 128) + norm + relu]
            *discriminator_block(128, 256),                             ## layer += [conv(128, 256) + norm + relu]
            *discriminator_block(256, 512),                             ## layer += [conv(256, 512) + norm + relu]
            nn.ZeroPad2d((1, 0, 1, 0)),                                 ## layer += [pad]
            nn.Conv2d(512, 1, 4, padding=1)                             ## layer += [conv(512, 1)]
        )

    def forward(self, img):             ## 输入(1, 3, 256, 256)
        return self.model(img)          ## 输出(1, 1, 16, 16)



# ## test
# img_shape = (3, 256, 256)
# n_residual_blocks = 9
# G_AB = GeneratorResNet(img_shape, n_residual_blocks)
# D_A = Discriminator(img_shape)
# img = torch.rand((1, 3, 256, 256))
# fake = G_AB(img)
# print(fake.shape)

# fake_D = D_A(img)
# print(fake_D.shape)
