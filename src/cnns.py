import sys
sys.path.append(".")

import torch as pt
import torch.nn as nn
import torch.nn.functional as F

ksize = 7
class DoubleConv(nn.Module):
    """(Conv => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.ReLU(inplace=True)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ksize, padding=(ksize-1)//2),
            nn.BatchNorm2d(mid_channels),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=ksize, padding=(ksize-1)//2),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=nn.LeakyReLU(0.2, inplace=True))
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(ksize+1)//2, stride=(ksize+1)//2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # If you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = pt.cat([x2, x1], dim=1)
        return self.dropout(self.conv(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_input, nf, n_output, dropout_rate=0.0, bilinear=True):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_input, nf)

        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        self.down3 = Down(nf * 4, nf * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(nf * 8, nf * 16 // factor)

        self.up1 = Up(nf * 16, nf * 8 // factor, dropout_rate, bilinear)
        self.up2 = Up(nf * 8, nf * 4 // factor, dropout_rate, bilinear)
        self.up3 = Up(nf * 4, nf * 2 // factor, dropout_rate, bilinear)
        self.up4 = Up(nf * 2, nf, dropout_rate, bilinear)

        self.outc = OutConv(nf, n_output)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class UNetShallow(nn.Module):
    def __init__(self, n_input, nf, n_output, dropout_rate=0.0, bilinear=True):
        super(UNetShallow, self).__init__()

        self.inc = DoubleConv(n_input, nf)

        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        factor = 2 if bilinear else 1
        self.down3 = Down(nf * 4, nf * 8 // factor)

        self.up1 = Up(nf * 8, nf * 4 // factor, dropout_rate, bilinear)
        self.up2 = Up(nf * 4, nf * 2 // factor, dropout_rate, bilinear)
        self.up3 = Up(nf * 2, nf, dropout_rate, bilinear)

        self.outc = OutConv(nf, n_output)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)


