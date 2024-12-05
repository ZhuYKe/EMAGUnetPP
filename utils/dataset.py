#!/usr/bin/env python
# coding=utf-8
import random
import re
import sys
import cv2
import os
import glob

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrainLoader(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        return image, label

    def __len__(self):
        return len(self.imgs_path)

class EvalLoader(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        return image, label

    def __len__(self):
        return len(self.imgs_path)


class ResultLoader(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        return image

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    test_data_path = sys.argv[1]
    test_data = TrainLoader(test_data_path)
    train_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    train_data_size = len(test_data)
    print(train_data_size)
