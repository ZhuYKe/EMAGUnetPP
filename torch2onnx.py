#!/usr/bin/env python
# coding=utf-8
import os
import glob
import sys

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.onnx

class DoubleConv(nn.Module):       # 连续两次卷积模块
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 卷积：输入通道数，输出通道数，卷积核大小3  步长默认1 填充1
            nn.BatchNorm2d(out_channels),        # BN：数据归一化
            nn.ReLU(inplace=True),        # ReLU :激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):   # 下采样模块（下采样+卷积）
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化 池化核大小2
            DoubleConv(in_channels, out_channels)   # 两次卷积
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):   # 上采样模块（上采样+特征融合+卷积）
    def __init__(self, in_channels, out_channels, bilinear):  # 定义上采样方法，并进行两次卷积
        super().__init__()
        if bilinear:  # 双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上池化
        else:
            self.up = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)  # 反卷积
        self.conv = DoubleConv(in_channels, out_channels)  # 两次卷积

    def forward(self, x1, x2):  # 上采样、特征融合、卷积
        x1 = self.up(x1)  # x1接收上采样数据，x2接收特征融合数据
        diffy = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffx = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = f.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):  # 整合输出通道
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 卷积： 卷积核大小1

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512+256, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)
        self.up4 = Up(64+32, 32, bilinear)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.outc(x9)
        return output



if len(sys.argv) < 2:
    print("请按格式输入")
    print("<script_name> <input_pth> <oonx_model_path>")
    print("input_pth：权重文件")
    print("oonx_model_path：ONNX格式神经网络模型")
    sys.exit(1)
input_pth = sys.argv[1]
oonx_model_path = sys.argv[2]

weight_file_path = (input_pth)
device=torch.device("cpu")
net = UNet(in_channels=1, num_classes=1)
net.to(device)

net.load_state_dict(torch.load(weight_file_path, map_location=device))  # 加载Unet训练权重
net.eval()  # 测试模式

dummy_input = torch.randn(1, 1, 256, 256).to(device)  # 根据输入大小调整
output = net(dummy_input)

# 导出为 ONNX
torch.onnx.export(net, dummy_input, oonx_model_path, verbose=True, training=False, input_names=["input"], output_names=["output"], opset_version=11)

