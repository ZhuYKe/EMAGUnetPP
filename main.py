#!/usr/bin/env python
# coding=utf-8
import os
import torch
from train import train_unet
from predict import test_unet

from utils.seed import setup_seed

from MODEL.UNet import UNet
from MODEL.EMAGUNetPP import EMAG


# torch.cuda.set_device(0)
setup_seed(42)  # 设置随机种子方便复现

work_path = "/mnt/d/ZYK_WorkSpace/Unet"

# 【输入数据集地址（该路径下需要包含train、eval、test三个数据集文件夹）】
data_path = work_path + "/data_test"

# 【日志文件、权重文件、loss曲线图将保存于此路径】
log_save_path = work_path + "/logs_test"
# 【测试结果图像和结果数据将保存于此路径】
Test_result_save_path = work_path + "/predict_result"

#  *********** 【选择训练模式or测试模式】*********
model_choose = 'Train'
# model_choose = 'Test'

# ***********【选择网络模式】************
net_choose = 'Unet'
# net_choose = 'EMAG_Unet++'

# 【epoch、batch_size、初始lr、早停轮数设置】
epochs = 150  # 200
batch_size = 4  # 8
lr = 0.1
early_stop_set = 30  # 20

# 【输入图片通道数量1 网络分类类别1种】
in_channels = 1
num_classes = 1

# 选择设备，有cuda用cuda，没有就用cpu
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_choose == 'Train':

    Train_data_path = data_path + "/train/"  # 训练数据集地址
    Eval_data_path = data_path + "/eval/"  # 验证数据集地址

    net_dir = log_save_path + "/" + f"{net_choose}"
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    Log_txt_save_path = net_dir + "/logs.txt"  # 日志文件保存路径
    Log_pth_save_path = net_dir + "/best.pth"  # 权重文件保存路径
    Loss_picture_save_path = net_dir + "/figure.png"  # loss曲线图保存路径

    if net_choose == 'Unet':
        Net = UNet(in_channels=in_channels, num_classes=num_classes)
        Net.to(device=Device)
    if net_choose == 'EMAG_Unet++':
        Net = EMAG(in_channels=in_channels, num_classes=num_classes)
        Net.to(device=Device)

    train_unet(net=Net, device=Device, train_data_path=Train_data_path, eval_data_path=Eval_data_path,
               train_epochs=epochs, train_batch_size=batch_size,
               train_lr=lr, early_stop_number=early_stop_set,
               log_pth_save_path=Log_pth_save_path, log_txt_save_path=Log_txt_save_path,
               loss_picture_save_path=Loss_picture_save_path)

if model_choose == 'Test':

    if net_choose == 'Unet':
        Net = UNet(in_channels=in_channels, num_classes=num_classes)
        Net.to(device=Device)
    if net_choose == 'EMAG_Unet++':
        Net = EMAG(in_channels=in_channels, num_classes=num_classes)
        Net.to(device=Device)

    Log_pth_save_path = log_save_path + f"/{net_choose}/best.pth"
    txt_log_save_path = Test_result_save_path + f"/{net_choose}"
    if not os.path.exists(txt_log_save_path):
        os.mkdir(txt_log_save_path)

    test_data_path = data_path + "/test/"  # 验证数据集地址
    test_unet(net=Net, device=Device, pth_path=Log_pth_save_path, test_data_dir=test_data_path,
              txt_log_save_dir=txt_log_save_path)
