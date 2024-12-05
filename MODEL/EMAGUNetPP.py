import torch
import torch.nn as nn
from thop import profile

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class EMAG(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.MaxPool = nn.MaxPool2d(2, 2)    # kernel_size = 2, stride = 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])  # 1  32
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 32  64
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 64  128
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])  # 128  256
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])  # 256  512

        self.EMA_Block1 = EMA(channels=32)
        self.EMA_Block2 = EMA(channels=64)
        self.EMA_Block3 = EMA(channels=128)
        self.EMA_Block4 = EMA(channels=256)
        self.EMA_Block5 = EMA(channels=512)

        self.conv0_1 = DoubleConv(nb_filter[0] * 1 + nb_filter[1], nb_filter[0])  # 96 32
        self.conv1_1 = DoubleConv(nb_filter[1] * 1 + nb_filter[2], nb_filter[1])  # 192 64
        self.conv2_1 = DoubleConv(nb_filter[2] * 1 + nb_filter[3], nb_filter[2])  # 384 128
        self.conv3_1 = DoubleConv(nb_filter[3] * 1 + nb_filter[4], nb_filter[3])  # 768 256

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0(x)
        x0_0_1 = self.EMA_Block1(x0_0_0)
        x0_0_2 = self.MaxPool(x0_0_1)

        x1_0_0 = self.conv1_0(x0_0_2)
        x1_0_1 = self.EMA_Block2(x1_0_0)
        x1_0_2 = self.MaxPool(x1_0_1)

        x2_0_0 = self.conv2_0(x1_0_2)
        x2_0_1 = self.EMA_Block3(x2_0_0)
        x2_0_2 = self.MaxPool(x2_0_1)

        x3_0_0 = self.conv3_0(x2_0_2)
        x3_0_1 = self.EMA_Block4(x3_0_0)
        x3_0_2 = self.MaxPool(x3_0_1)

        x4_0_0 = self.conv4_0(x3_0_2)

        x0_1 = self.conv0_1(torch.cat([x0_0_0, self.up(x1_0_0)], 1))
        x0_1 = self.EMA_Block1(x0_1)
        x1_1 = self.conv1_1(torch.cat([x1_0_0, self.up(x2_0_0)], 1))
        x1_1 = self.EMA_Block2(x1_1)
        x2_1 = self.conv2_1(torch.cat([x2_0_0, self.up(x3_0_0)], 1))
        x2_1 = self.EMA_Block3(x2_1)
        x3_1 = self.conv3_1(torch.cat([x3_0_0, self.up(x4_0_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.EMA_Block1(x0_2)
        x1_2 = self.conv1_2(torch.cat([x1_0_0, x1_1, self.up(x2_1)], 1))
        x1_2 = self.EMA_Block2(x1_2)
        x2_2 = self.conv2_2(torch.cat([x2_0_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.EMA_Block1(x0_3)
        x1_3 = self.conv1_3(torch.cat([x1_0_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


if __name__ == '__main__':
    # unet = EMA_UNetPP_A4(in_channels=1, num_classes=1)
    # print(unet)
    # total_params = sum(p.numel() for p in unet.parameters())
    # print(f"Total Parameters: {total_params}")  # 9228349
    model = EMAG(in_channels=1, num_classes=1)
    model.eval()
    image = torch.randn(4, 1, 256, 256)
    flops, params = profile(model, inputs=(image,))
    print(f"{(flops / 1000000000.0)}G", params)
