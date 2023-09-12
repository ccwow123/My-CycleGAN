import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils.spectral_norm as spectral_norm

from utils.src.mod import CBAM,NONLocalBlock2D
from utils.src.SPADE_net import SPADE

# from src.mod import CBAM,NONLocalBlock2D
# from src.SPADE_net import SPADE

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# 加入CBAM模块在skip connection中
class GeneratorUNet_A(GeneratorUNet):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(in_channels, out_channels)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, self.cbam4(d7))
        u2 = self.up2(u1, self.cbam4(d6))
        u3 = self.up3(u2, self.cbam4(d5))
        u4 = self.up4(u3, self.cbam4(d4))
        u5 = self.up5(u4, self.cbam3(d3))
        u6 = self.up6(u5, self.cbam2(d2))
        u7 = self.up7(u6, self.cbam1(d1))

        return self.final(u7)


# 加入CBAM模块在encoder中
class GeneratorUNet_A_en(GeneratorUNet_A):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(self.cbam1(d1))
        d3 = self.down3(self.cbam2(d2))
        d4 = self.down4(self.cbam3(d3))
        d5 = self.down5(self.cbam4(d4))
        d6 = self.down6(self.cbam4(d5))
        d7 = self.down7(self.cbam4(d6))
        d8 = self.down8(self.cbam4(d7))
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)



# 加入SPADE
class SPADEGenerator(GeneratorUNet):
    def __init__(self, in_channels=3, out_channels=3,style_size=3):
        super().__init__(in_channels, out_channels)
        self.norm1 = SPADE(64,style_size)
        self.norm2 = SPADE(128, style_size)
        self.norm3 = SPADE(256, style_size)
        self.norm4 = SPADE(512, style_size)
        self.norm5 = SPADE(512, style_size)
        self.norm6 = SPADE(512, style_size)
        self.norm7 = SPADE(512, style_size)

        self.norm8 = SPADE(1024, style_size)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.norm1(d1, x))
        d3 = self.down3(self.norm2(d2, x))
        d4 = self.down4(self.norm3(d3, x))
        d5 = self.down5(self.norm4(d4, x))
        d6 = self.down6(self.norm5(d5, x))
        d7 = self.down7(self.norm6(d6, x))
        d8 = self.down8(self.norm7(d7, x))
        u1 = self.up1(d8, d7)
        u2 = self.up2(self.norm8(u1, x), d6)
        u3 = self.up3(self.norm8(u2, x), d5)
        u4 = self.up4(self.norm8(u3, x), d4)
        u5 = self.up5(self.norm8(u4, x), d3)
        u6 = self.up6(self.norm4(u5, x), d2)
        u7 = self.up7(self.norm3(u6, x), d1)

        return self.final(self.norm2(u7, x))

# 加入Non-local
class NL_Generator(GeneratorUNet):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(in_channels, out_channels)
        self.nl1 = NONLocalBlock2D(64)
        self.nl2 = NONLocalBlock2D(128)
        self.nl3 = NONLocalBlock2D(256)
        self.nl4 = NONLocalBlock2D(512)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.nl1(d1))
        d3 = self.down3(self.nl2(d2))
        d4 = self.down4(self.nl3(d3))
        d5 = self.down5(self.nl4(d4))
        d6 = self.down6(self.nl4(d5))
        d7 = self.down7(self.nl4(d6))
        d8 = self.down8(self.nl4(d7))
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)





##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3,height=256,width=256,n_conv=4):
        super(Discriminator, self).__init__()
        self.output_shape = (1, height // 2 ** n_conv, width // 2 ** n_conv)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# 卷积核大小为3
class Discriminator2(nn.Module):
    def __init__(self, in_channels=3,height=256,width=256,n_conv=5):
        super().__init__()
        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** n_conv, width // 2 ** n_conv)

        def discriminator_block(in_filters, out_filters, normalization=True, padding=1):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=padding)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv1 = nn.Sequential(*discriminator_block(in_channels * 2, 64, normalization=False))
        self.conv2 = nn.Sequential(*discriminator_block(64, 128))
        self.conv3 = nn.Sequential(*discriminator_block(128, 256))
        self.conv4 = nn.Sequential(*discriminator_block(256, 512))
        self.conv5 = nn.Sequential(*discriminator_block(512, 1024))


        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.out = nn.Conv2d(1024, 1, 3, padding=1, bias=False)
        # self.model = nn.Sequential(
        #     *discriminator_block(in_channels * 2, 64, normalization=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        #     *discriminator_block(512, 1024),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(1024, 1, 4, padding=1, bias=False)
        # )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        x = self.conv1(img_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

        # return self.model(img_input)

# 加入谱范数归一化SN
class Discriminator_SN(Discriminator):
    def __init__(self, in_channels=3,height=256,width=256,n_conv=4):
        super().__init__(in_channels,height,width,n_conv)
        def Conv_SN_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *Conv_SN_block(in_channels * 2, 64, normalization=False),
            *Conv_SN_block(64, 128),
            *Conv_SN_block(128, 256),
            *Conv_SN_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, 4, padding=1, bias=False))
        )


if __name__ == "__main__":
    model = NL_Generator()
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print(out.shape)