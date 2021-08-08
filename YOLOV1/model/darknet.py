import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d

"""
Darknet is based on architecture of the paper
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Darknet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(Darknet, self).__init__()
        self.conv_1 = ConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2 = ConvBlock(
            in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3 = nn.Sequential(
            ConvBlock(in_channels=192, out_channels=128,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
        )
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),

            ConvBlock(in_channels=512, out_channels=512,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=1, padding=1)
        )
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2)

        self.conv_5 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=512,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=1024, out_channels=512,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=1, padding=1),

            ConvBlock(in_channels=1024, out_channels=1024,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=1024, out_channels=1024,
                      kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        x = self.conv_5(x)
        return x


def test():
    x = torch.randn(1, 3, 448, 448)
    darknet = Darknet()
    print(darknet(x).shape)


if __name__ == '__main__':
    test()
