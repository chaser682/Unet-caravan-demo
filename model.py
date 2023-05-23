"""
@Author : Chaser
@Time : 2023/3/3 14:50
@File : dataload.py
@software : PyCharm
"""
"""
@update:
2023/3/5
2023/3/10
2023/5/10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# print(torch.__version__)


# 连续两次卷积过程
class DoubleConv(nn.Module):
    '''Convolution -> BN -> ReLU'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # 经过卷积后，输出层h/w等于输入层
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样过程
class Down(nn.Module):
    """Maxpool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 上采样过程
class Up(nn.Module):
    """ConvolutionTranspose -> DoubleConv"""

    def __init__(self, in_channels, out_channels, transpose=False):
        super(Up, self).__init__()

        '''conv: out_shape = (in_shape + 2*padding - kernel_size)/stride + 1
           conv_transpose: out_shape = (in_shape - 1)*strde + kernel_size - 2*padding
        '''
        if transpose:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=2, padding=0),
                nn.ReLU(inplace=True)
            )
        self.conv = DoubleConv(in_channels, out_channels)
        # 传递参数
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):
        """
        x1为上采样单元，x2为与x1同层的下采样单元
        """

        x1 = self.up(x1)

        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2)  # 分别表示上下左右
                   )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    # 设置参数权重
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

# 最后一次卷积得到输出
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Unet网络结构
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        # 输入卷积
        self.inc = DoubleConv(in_channels, 64)

        # 下采样
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        # 最后两次下采样进行dropout操作，防止过拟合
        self.down3 = Down(256, 512)
        self.drop3 = nn.Dropout2d(0.3)
        self.down4 = Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.4)

        # 上采样
        self.up1 = Up(1024, 512, False)
        self.up2 = Up(512, 256, False)
        self.up3 = Up(256, 128, False)
        self.up4 = Up(128, 64, False)

        # 得到卷积输出
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

if __name__ == "__main__":
    model = Unet(in_channels=3,out_channels=1)

    inp = torch.randn(1,3,224,224)
    outp = model(inp)
    print(outp)