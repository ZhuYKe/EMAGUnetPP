#!/usr/bin/env python
# coding=utf-8
import os

import torch
from torch.utils.data import DataLoader

from MODEL.UNet import UNet


from utils.dataset import EvalLoader
from utils.score import iou_score, TPR_score, PPV_score


# Unet
def test_unet(net, device, pth_path, test_data_dir, txt_log_save_dir):
    test_data = EvalLoader(test_data_dir)  # 测试集地址传入dataset_EvalLoader对image、label进行数据预处理
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net.load_state_dict(torch.load(pth_path, map_location=device))  # 加载Unet训练权重

    net.eval()  # 测试模式

    i = 0  # 读图像计数 初始化为0
    total_iou = 0  # 测试集评价指标iou总和，除以i后为测试集平均iou
    total_PPV = 0
    total_TPR = 0

    # 遍历所有图片
    for image, label in test_dataloader:
        image = image.to(device=device, dtype=torch.float32)  # 将数据拷贝到device中
        label = label.to(device=device, dtype=torch.float32)

        i = i + 1

        output = net(image)  # 预测

        iou = iou_score(output, label)  # 评价指标
        total_iou = total_iou + iou  # 更新iou总和

        PPV = PPV_score(output, label)
        total_PPV = total_PPV + PPV

        TPR = TPR_score(output, label)
        total_TPR = total_TPR + TPR

    print("测试集iou : {} , 测试集PPV ：{} , 测试集TPR ：{} ".format((total_iou / i), (total_PPV / i), (total_TPR / i)))  # 打印测试集整体平均iou

    # 文件名
    output_file_name = 'log.txt'

    # 文件路径
    output_file_path = os.path.join(txt_log_save_dir, output_file_name)

    # 写入文件
    with open(output_file_path, 'w') as file:
        file.write("Iou: {} ".format((total_iou / i)))
        file.write("PPV: {} ".format((total_PPV / i)))
        file.write("TPR: {} ".format((total_TPR / i)))

    print(f"评估指标已保存至{output_file_path}")
