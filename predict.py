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
    test_data = EvalLoader(test_data_dir)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net.load_state_dict(torch.load(pth_path, map_location=device))

    net.eval()

    i = 0
    total_iou = 0
    total_PPV = 0
    total_TPR = 0

    for image, label in test_dataloader:
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        i = i + 1

        output = net(image)

        iou = iou_score(output, label)
        total_iou = total_iou + iou

        PPV = PPV_score(output, label)
        total_PPV = total_PPV + PPV

        TPR = TPR_score(output, label)
        total_TPR = total_TPR + TPR

    print("TestDataset_iou : {} , TestDataset_PPV ：{} , TestDataset_TPR ：{} ".format((total_iou / i), (total_PPV / i), (total_TPR / i)))

    output_file_name = 'log.txt'

    output_file_path = os.path.join(txt_log_save_dir, output_file_name)

    with open(output_file_path, 'w') as file:
        file.write("Iou: {} ".format((total_iou / i)))
        file.write("PPV: {} ".format((total_PPV / i)))
        file.write("TPR: {} ".format((total_TPR / i)))

    print(f"log Save as {output_file_path}")
