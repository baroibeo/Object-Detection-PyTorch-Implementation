import torch
import torch.nn as nn
from darknet import ConvBlock, Darknet


class YoloV1(nn.Module):
    def __init__(self, _darknet, num_classes, grid_size, num_predboxes):
        """
        num_predboxes : predicted boxes per cell 
        each cell has shape : grid_size x grid_size
        """
        super(YoloV1, self).__init__()
        self.grid_size = grid_size
        self.num_predboxes = num_predboxes
        self.num_classes = num_classes
        self.darknet = _darknet
        self.conv = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=1024,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=1024, out_channels=1024,
                      kernel_size=3, stride=1, padding=1)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3, inplace=False),
            nn.Linear(4096, grid_size*grid_size*(5*num_predboxes+num_classes))
        )

    def forward(self, x):
        x = self.darknet(x)
        x = self.conv(x)
        x = self.linear(x)
        x = x.view(-1, self.grid_size, self.grid_size, 5 *
                   self.num_predboxes+self.  num_classes)
        return x


def test():
    x = torch.randn(1, 3, 448, 448)
    darknet = Darknet(in_channels=3, out_channels=1024)
    yolov1 = YoloV1(darknet, num_classes=20, grid_size=7, num_predboxes=2)
    print(yolov1(x).shape)


if __name__ == '__main__':
    test()
