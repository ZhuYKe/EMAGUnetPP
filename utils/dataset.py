#!/usr/bin/env python
# coding=utf-8
import random
import re
import cv2
import os
import glob

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrainLoader_new(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs1_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs1_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs2_path = glob.glob(os.path.join(data_path, 'image_ph2/*.png'))
        self.imgs2_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs3_path = glob.glob(os.path.join(data_path, 'image_ph3/*.png'))
        self.imgs3_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.ipgs_path = glob.glob(os.path.join(data_path, 'image_ipgs/*.tif'))
        self.ipgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.stacking_path = glob.glob(os.path.join(data_path, 'image_stacking/*.tif'))
        self.stacking_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片1
        image1_path = self.imgs1_path[index]

        # 根据image_path生成label_path
        label_path = image1_path.replace('image', 'label')

        # 读取训练图片1.png和标签图片.png(三通道 [0-255] uint8)
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1])

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        # # 根据index读取图片2.png(三通道 [0-255] uint8)
        # image2_path = self.imgs2_path[index]
        # # 读取训练图片2
        # image2 = cv2.imread(image2_path)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # image2 = image2.reshape(1, image2.shape[0], image2.shape[1]).astype(np.float32)

        # # 根据index读取图片3.png(三通道 [0-255] uint8)
        # image3_path = self.imgs3_path[index]
        # # 读取训练图片3
        # image3 = cv2.imread(image3_path)
        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        # image3 = image3.reshape(1, image3.shape[0], image3.shape[1]).astype(np.float32)

        # # 根据index读取图片ipgs.tif(单波段 [0-1] float32)
        # ipgs_path = self.ipgs_path[index]
        # # 读取训练图片ipgs
        # ipgs = cv2.imread(ipgs_path, cv2.IMREAD_UNCHANGED)
        # ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)

        # # 根据index读取图片ipgs.tif(单波段 [0-255] uint8)
        # ipgs_path = self.ipgs_path[index]
        # ipgs = cv2.imread(ipgs_path)
        # ipgs = cv2.cvtColor(ipgs, cv2.COLOR_BGR2GRAY)
        # ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)
        #
        # # 根据index读取图片stacking.tif(单波段 [?-?] float32)
        # stacking_path = self.stacking_path[index]
        # # 读取训练图片ipgs
        # stacking = cv2.imread(stacking_path, cv2.IMREAD_UNCHANGED)
        # stacking = stacking.reshape(1, stacking.shape[0], stacking.shape[1])

        # combined_image = np.concatenate((image1, image2, image3, ipgs, stacking), axis=0)
        combined_image = image1

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = (label / 255)

        return combined_image, label

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs1_path)


class EvalLoader_new(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs1_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs1_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs2_path = glob.glob(os.path.join(data_path, 'image_ph2/*.png'))
        self.imgs2_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs3_path = glob.glob(os.path.join(data_path, 'image_ph3/*.png'))
        self.imgs3_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.ipgs_path = glob.glob(os.path.join(data_path, 'image_ipgs/*.tif'))
        self.ipgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.stacking_path = glob.glob(os.path.join(data_path, 'image_stacking/*.tif'))
        self.stacking_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片1
        image1_path = self.imgs1_path[index]

        # 根据image_path生成label_path
        label_path = image1_path.replace('image', 'label')

        # 读取训练图片1.png和标签图片.png(三通道 [0-255] uint8)
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1])

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        # # 根据index读取图片2.png(三通道 [0-255] uint8)
        # image2_path = self.imgs2_path[index]
        # # 读取训练图片2
        # image2 = cv2.imread(image2_path)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # image2 = image2.reshape(1, image2.shape[0], image2.shape[1]).astype(np.float32)

        # # 根据index读取图片3.png(三通道 [0-255] uint8)
        # image3_path = self.imgs3_path[index]
        # # 读取训练图片3
        # image3 = cv2.imread(image3_path)
        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        # image3 = image3.reshape(1, image3.shape[0], image3.shape[1]).astype(np.float32)

        # # 根据index读取图片ipgs.tif(单波段 [0-1] float32)
        # ipgs_path = self.ipgs_path[index]
        # # 读取训练图片ipgs
        # ipgs = cv2.imread(ipgs_path, cv2.IMREAD_UNCHANGED)
        # ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)

        # # 根据index读取图片ipgs.tif(单波段 [0-255] uint8)
        # ipgs_path = self.ipgs_path[index]
        # ipgs = cv2.imread(ipgs_path)
        # ipgs = cv2.cvtColor(ipgs, cv2.COLOR_BGR2GRAY)
        # ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)
        #
        # # 根据index读取图片stacking.tif(单波段 [?-?] float32)
        # stacking_path = self.stacking_path[index]
        # # 读取训练图片ipgs
        # stacking = cv2.imread(stacking_path, cv2.IMREAD_UNCHANGED)
        # stacking = stacking.reshape(1, stacking.shape[0], stacking.shape[1])

        # combined_image = np.concatenate((image1, image2, image3, ipgs, stacking), axis=0)
        combined_image = image1

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = (label / 255)

        return combined_image, label

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs1_path)


class ResultLoader_new(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs1_path = glob.glob(os.path.join(data_path, 'image_ph1/*.png'))
        self.imgs1_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs2_path = glob.glob(os.path.join(data_path, 'image_ph2/*.png'))
        self.imgs2_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.imgs3_path = glob.glob(os.path.join(data_path, 'image_ph3/*.png'))
        self.imgs3_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.ipgs_path = glob.glob(os.path.join(data_path, 'image_ipgs/*.tif'))
        self.ipgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

        self.stacking_path = glob.glob(os.path.join(data_path, 'image_stacking/*.tif'))
        self.stacking_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片1
        image1_path = self.imgs1_path[index]
        # 读取训练图片1.png(三通道 [0-255] uint8)
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1]).astype(np.float32)

        # 根据index读取图片2.png(三通道 [0-255] uint8)
        image2_path = self.imgs2_path[index]
        # 读取训练图片2
        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1]).astype(np.float32)

        # 根据index读取图片3.png(三通道 [0-255] uint8)
        image3_path = self.imgs3_path[index]
        # 读取训练图片3
        image3 = cv2.imread(image3_path)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        image3 = image3.reshape(1, image3.shape[0], image3.shape[1]).astype(np.float32)

        # # 根据index读取图片ipgs.tif(单波段 [0-1] float32)
        # ipgs_path = self.ipgs_path[index]
        # # 读取训练图片ipgs
        # ipgs = cv2.imread(ipgs_path, cv2.IMREAD_UNCHANGED)
        # ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)

        # 根据index读取图片ipgs.tif(单波段 [0-1] uint8)
        ipgs_path = self.ipgs_path[index]
        ipgs = cv2.imread(ipgs_path)
        ipgs = cv2.cvtColor(ipgs, cv2.COLOR_BGR2GRAY)
        ipgs = ipgs.reshape(1, ipgs.shape[0], ipgs.shape[1]).astype(np.float32)

        # 根据index读取图片stacking.tif(单波段 [?-?] float32)
        stacking_path = self.stacking_path[index]
        # 读取训练图片ipgs
        stacking = cv2.imread(stacking_path, cv2.IMREAD_UNCHANGED)
        stacking = stacking.reshape(1, stacking.shape[0], stacking.shape[1])

        combined_image = np.concatenate((image1, image2, image3, ipgs, stacking), axis=0)

        return combined_image

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs1_path)



class TrainLoader(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        return image, label

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs_path)

class EvalLoader(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(1, label.shape[0], label.shape[1])

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        return image, label

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs_path)


class ResultLoader(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
        self.imgs_path.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 对图片路径按照数字进行排序

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 读取图片
        image = cv2.imread(image_path)

        # 将数据转为单通道图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])

        return image

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    test_data_path = 'D:/ZYK/Unet/data/chenxianmoni/train/'  # 测试图像存储地址
    test_data = TrainLoader(test_data_path)  # 测试集地址传入dataset_EvalLoader进行数据预处理
    train_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    train_data_size = len(test_data)  # 获取训练集数据大小
    print(train_data_size)
