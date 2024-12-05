#!/usr/bin/env python
# coding=utf-8
import os
import torch
from train import train_unet
from predict import test_unet

from utils.seed import setup_seed

from MODEL.UNet import UNet
from MODEL.EMAGUNetPP import EMAG


setup_seed(42)  # Fixed random seed

work_path = "/mnt/d/ZYK_WorkSpace/Unet"
data_path = work_path + "/data_test"
log_save_path = work_path + "/logs_test"
Test_result_save_path = work_path + "/predict_result"

#  *********** 【Select training mode or test mode】*********
model_choose = 'Train'
# model_choose = 'Test'

# ***********【Choose MODEL】************
net_choose = 'Unet'
# net_choose = 'EMAG_Unet++'

# 【epoch、batch_size、ori_lr、earlystop】
epochs = 150  # 200
batch_size = 4  # 8
lr = 0.1
early_stop_set = 30  # 20

# 【Input picture Number of channels 1 Network category Category 1】
in_channels = 1
num_classes = 1

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_choose == 'Train':

    Train_data_path = data_path + "/train/"
    Eval_data_path = data_path + "/eval/"

    net_dir = log_save_path + "/" + f"{net_choose}"
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    Log_txt_save_path = net_dir + "/logs.txt"
    Log_pth_save_path = net_dir + "/best.pth"
    Loss_picture_save_path = net_dir + "/figure.png"

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

    test_data_path = data_path + "/test/"
    test_unet(net=Net, device=Device, pth_path=Log_pth_save_path, test_data_dir=test_data_path,
              txt_log_save_dir=txt_log_save_path)
