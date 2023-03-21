import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBlock(nn.Module):
    def __init__(self, input_nc,output_nc,mid_channels=None,stride=1):
        super().__init__()
        if mid_channels is None:
            mid_channels = output_nc
        conv_block = [  nn.Conv2d(input_nc, mid_channels, 3, stride=stride, padding=1),
                    nn.InstanceNorm2d(mid_channels),
                    nn.LeakyReLU(0.2, inplace=True)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BaseBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BaseBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # x = torch.cat([x2, x1], dim=1)
        x = x1 + x2
        x = self.conv(x)
        return x
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True):
        super(Generator, self).__init__()
        #           0, 1, 2, 3, 4
        channles = [32,64,128,256,512]
        self.con_1 = BaseBlock(input_nc,channles[0],stride=1)
        self.con_2 = BaseBlock(channles[0],channles[0],stride=2)
        self.con_3 = BaseBlock(channles[0],channles[1],stride=1)
        self.con_4 = BaseBlock(channles[1],channles[1],stride=2)
        self.con_5 = BaseBlock(channles[1],channles[2],stride=1)
        self.con_6 = BaseBlock(channles[2],channles[2],stride=2)
        self.con_7 = BaseBlock(channles[2],channles[3],stride=1)
        self.con_8 = BaseBlock(channles[3],channles[3],stride=2)
        self.con_9 = BaseBlock(channles[3],channles[4],stride=1)
        self.con_10 = BaseBlock(channles[4],channles[4],stride=2) #bottleneck


        self.up_9 = Up(channles[4],channles[3])
        self.decon_8 = BaseBlock(channles[3],channles[3],stride=1)
        self.up7 = Up(channles[3],channles[2])
        self.decon_6 = BaseBlock(channles[2],channles[2],stride=1)
        self.up5 = Up(channles[2],channles[1])
        self.decon_4 = BaseBlock(channles[1],channles[1],stride=1)
        self.up3 = Up(channles[1],channles[0])
        self.decon_2 = BaseBlock(channles[0],channles[0],stride=1)
        self.up1 = Up(channles[0],channles[0])


        # self.out = nn.Conv2d(channles[0], output_nc, kernel_size=1, stride=1, padding=0)
        out= [  nn.ReflectionPad2d(3),
                    nn.Conv2d(channles[0], output_nc, 7),
                    nn.Tanh() ]

        self.out = nn.Sequential(*out)
    def forward(self, x):
        x1 = self.con_1(x)
        x2 = self.con_2(x1)
        x3 = self.con_3(x2)
        x4 = self.con_4(x3)
        x5 = self.con_5(x4)
        x6 = self.con_6(x5)
        x7 = self.con_7(x6)
        x8 = self.con_8(x7)
        x9 = self.con_9(x8)
        x10 = self.con_10(x9)

        x9_2 = self.up_9(x10,x9)
        x8_2 = self.decon_8(x9_2)
        x7_2 = self.up7(x8_2,x7)
        x6_2 = self.decon_6(x7_2)
        x5_2 = self.up5(x6_2,x5)
        x4_2 = self.decon_4(x5_2)
        x3_2 = self.up3(x4_2,x3)
        x2_2 = self.decon_2(x3_2)
        x1_2 = self.up1(x2_2,x1)
        x = self.out(x1_2)


        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        #          0, 1, 2, 3
        channels = [64,128,256,512]
        con_1 = [   nn.Conv2d(input_nc, channels[0], 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        self.con_1 = nn.Sequential(*con_1)
        self.con_2 = BaseBlock(channels[0],channels[1],stride=2)
        self.con_3 = BaseBlock(channels[1],channels[2],stride=2)
        self.con_4 = BaseBlock(channels[2],channels[3],stride=2)
        self.con_5 = BaseBlock(channels[3],channels[3],stride=1)
        self.out = nn.Conv2d(channels[3], 1, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.con_1(x)
        x = self.con_2(x)
        x = self.con_3(x)
        x = self.con_4(x)
        x = self.con_5(x)
        x = self.out(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == '__main__':
    # Test the model
    # x = torch.randn((1, 3, 512, 512))
    # G = Generator(3, 1)
    # y = G(x)
    # print('y.shape:' , y.shape)

    y = torch.randn((1, 3, 512, 512))
    D = Discriminator(3)
    out = D(y)
    print(out.shape)