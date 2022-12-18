import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
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
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

##############################
#           Generators
##############################


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 16, normalize=False)
        self.down2 = UNetDown(16, 32, dropout=0.5)
        self.down3 = UNetDown(32, 64, dropout=0.5)
        self.down4 = UNetDown(64, 64, normalize=False, dropout=0.5)

        self.up1 = UNetUp(64, 64, dropout=0.5)
        self.up2 = UNetUp(128, 32, dropout=0.5)
        self.up3 = UNetUp(64, 16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        return self.final(u3)

# class Generator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(Generator, self).__init__()
#         self.down1 = UNetDown(in_channels, 16, normalize=False)
#         self.down2 = UNetDown(16, 32, dropout=0.5)
#         self.down3 = UNetDown(32, 64, dropout=0.5)
#         self.down4 = UNetDown(64, 128, normalize=False, dropout=0.5)
#         self.down5 = UNetDown(128, 128, normalize=False, dropout=0.5)
#
#         self.up1 = UNetUp(128, 128, dropout=0.5)
#         self.up2 = UNetUp(256, 64, dropout=0.5)
#         self.up3 = UNetUp(128, 32, dropout=0.5)
#         self.up4 = UNetUp(64, 16)
#
#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(32, out_channels, 4, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         # U-Net generator with skip connections from encoder to decoder
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)
#
#         u1 = self.up1(d5, d4)
#         u2 = self.up2(u1, d3)
#         u3 = self.up3(u2, d2)
#         u4 = self.up4(u3, d1)
#
#         return self.final(u4)
##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, normalization=False),
            *discriminator_block(16, 32),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)



