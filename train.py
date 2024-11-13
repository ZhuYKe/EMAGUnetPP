#!/usr/bin/env python
# coding=utf-8
from torch.utils.data import DataLoader
from utils.dataset import TrainLoader
from utils.dataset import EvalLoader
from utils.dataset import TrainLoader_new
from utils.dataset import EvalLoader_new
from utils.score import iou_score, pixel_accuracy
from utils.draw import draw
from torch import optim
import torch.nn as nn
import torch
import time
from tqdm import tqdm

def train_unet(net, device,
               train_data_path,
               eval_data_path,
               train_epochs,
               train_batch_size,
               train_lr,
               early_stop_number,
               log_pth_save_path,
               log_txt_save_path,
               loss_picture_save_path):

    # train_data = TrainLoader(train_data_path)    # 训练集地址传入dataset_TrainLoader进行数据预处理
    # eval_data = EvalLoader(eval_data_path)  # 验证集地址传入dataset_EvalLoader进行数据预处理

    train_data = TrainLoader_new(train_data_path)    # 训练集地址传入dataset_TrainLoader进行数据预处理
    eval_data = EvalLoader_new(eval_data_path)  # 验证集地址传入dataset_EvalLoader进行数据预处理

    print("本次训练 epoch:{} batch_size:{}  初始学习率：{}".format(train_epochs, train_batch_size, train_lr))

    # dataloader加载处理后训练、测试数据集
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_data, batch_size=train_batch_size, shuffle=True)

    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=train_lr, betas=(0.9, 0.99))  # 算法：Adam
    # optimizer = optim.SGD(net.parameters(),lr=train_lr,momentum=0.8)  # 算法：Momentum
    # optimizer = optim.RMSprop(net.parameters(), lr=train_lr, alpha=0.9)    # 算法：RMSprop

    # 定义学习率衰减策略
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9, 15, 22], gamma=0.1)  # 阶梯调整
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)  # 指数下降 0.95

    # 定义损失函数
    loss_fn = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数
    # loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

    best_loss = float('inf')  # 记录最佳验证loss，初始化为正无穷
    best_iou = 0
    early_stop_counter = 0  # 早停计数
    training_time = 0  # 训练总时间计时

    train_data_size = len(train_data)  # 获取训练集数据大小
    eval_data_size = len(eval_data)  # 获取验证集数据大小

    # 训练参数可视化
    print("训练集：{}张   验证集：{}张".format(train_data_size, eval_data_size))
    print("优化算法：Adam 学习率衰减策略：指数下降 损失函数：二分类交叉熵")

    # 清空之前内容 并写入新的训练日志标题
    output_txt = "EPOCH: Train_Loss: Eval_Loss: Eval_iou: Eval_PA:"
    with open(log_txt_save_path, "a+") as f:
        f.truncate(0)
        f.write(output_txt + '\n')
        f.close()

    # 开始训练、验证循环
    for train_epoch in range(train_epochs):    # 训练epochs次

        print("===================== EPOCH {}/{} =====================".format((train_epoch+1), train_epochs))

        net.train()        # 训练模式标志
        total_train_loss = 0  # 统计每轮epoch的训练loss总量，每轮开始前初始化为0
        total_train_iou = 0  # 统计每轮epoch的训练iou总量，每轮开始前初始化为0
        total_train_PA = 0  # 统计每轮epoch的训练PA总量，每轮开始前初始化为0
        start = time.time()  # 每轮epoch训练开始时间计时

        # 训练步骤开始
        print("训练中")
        train_pbar = tqdm(total=train_data_size)  # 训练进度条开启
        for image, label in train_dataloader:        # 按照batch_size开始训练

            image = image.to(device=device, dtype=torch.float32)            # 将数据拷贝到device中
            label = label.to(device=device, dtype=torch.float32)

        # Unet 计算loss
            output = net(image)  # 使用网络参数，输出预测结果
            loss = loss_fn(output, label)  # 计算此次传播loss
            iou = iou_score(output, label)  # 计算此次传播iou
            PA = pixel_accuracy(output, label)  # 计算此次传播PA

            optimizer.zero_grad()  # 优化前梯度清零
            loss.backward()            # 调用损失的反向传播，得到每个参数检验的梯度
            optimizer.step()  # 调用优化器

            total_train_loss = total_train_loss + loss.item()   # 计算每轮epoch训练loss总量
            total_train_iou = total_train_iou + iou  # 计算每轮epoch训练iou总量
            total_train_PA = total_train_PA + PA  # 计算每轮epoch训练iou总量

            time.sleep(0.05)
            train_pbar.update(1 * train_batch_size)
        train_pbar.close()  # 训练进度条关闭
        # 训练步骤结束

        # 记录训练总时长
        end = time.time()
        time_elapsed = end - start
        training_time = training_time + time_elapsed

        # 打印此轮epoch平均训练loss iou
        print("Train_Loss={}   Train_iou={}   Train_PA={}".format(
            ((total_train_loss * train_batch_size) / train_data_size),
            ((total_train_iou * train_batch_size) / train_data_size),
            ((total_train_PA * train_batch_size) / train_data_size)))

        net.eval()  # 验证模式标志
        total_eval_loss = 0  # 统计每轮epoch的验证loss总量，每轮开始前初始化为0
        total_eval_iou = 0  # 统计每轮epoch的验证iou总量，每轮开始前初始化为0
        total_eval_PA = 0  # 统计每轮epoch的验证PA总量，每轮开始前初始化为0

        # 验证步骤开始
        with torch.no_grad():  # 验证部分无反向传播不参与训练
            print("验证中")
            for image, label in eval_dataloader:

                image = image.to(device=device, dtype=torch.float32)  # 将数据拷贝到device中
                label = label.to(device=device, dtype=torch.float32)

                # Unet 计算loss
                output = net(image)  # 使用网络参数，输出预测结果
                loss = loss_fn(output, label)  # 计算此次验证loss
                iou = iou_score(output, label)  # 计算此次验证iou
                PA = pixel_accuracy(output, label)

                total_eval_loss = total_eval_loss + loss.item()  # 计算每轮epoch验证loss总量
                total_eval_iou = total_eval_iou + iou  # 计算每轮epoch验证iou总量
                total_eval_PA = total_eval_PA + PA  # 计算每轮epoch验证iou总量
        # 验证步骤结束

        # 打印此轮epoch平均验证loss iou
        print("Eval_Loss={}   Eval_iou={}   Eval_PA={}".format(
            ((total_eval_loss * train_batch_size) / eval_data_size),
            ((total_eval_iou * train_batch_size) / eval_data_size),
            ((total_eval_PA * train_batch_size) / eval_data_size)))

        # 学习率随epoch更新
        scheduler.step(train_epoch)

        # 将loss、iou等写入日志
        output_txt = "%d %f %f %f %f" % ((train_epoch + 1),
                                      ((total_train_loss * train_batch_size) / train_data_size),
                                      ((total_eval_loss * train_batch_size) / eval_data_size),
                                      ((total_eval_iou * train_batch_size) / eval_data_size),
                                      ((total_eval_PA * train_batch_size) / eval_data_size))
        with open(log_txt_save_path, "a+") as f:
            f.write(output_txt + '\n')
            f.close()

        # 根据验证指标保存权重文件
        if total_eval_iou > best_iou:
            best_iou = total_eval_iou
        # if total_eval_loss < best_loss:  # 若验证集loss降低，更新best_loss，保存权重文件
        #     best_loss = total_eval_loss
            torch.save(net.state_dict(), log_pth_save_path)  # 刷新此轮epoch的权重文件
            print("========【EPOCH {} 此轮模型已刷新】========".format(train_epoch+1))
            early_stop_counter = 0  # 验证集出现更低loss，说明未过拟合，早停计数归零
        else:
            print("========【EPOCH {} 此轮模型未刷新】========".format(train_epoch + 1))
            early_stop_counter = early_stop_counter + 1  # 验证集loss不下降，早停计数+1

        #  连续x轮epoch的loss未降低，可能出现过拟合，进行早停，结束训练
        if early_stop_counter >= early_stop_number:
            print("==================== Early Stopping ====================")
            break

    # 打开txt对loss进行可视化
    draw(log_txt_save_path, loss_picture_save_path)

    # 将训练参数录入日志
    output_txt = "本次训练 Epoch:%d batchsize:%d 初始学习率:%f" % (train_epochs, train_batch_size, train_lr)
    with open(log_txt_save_path, "a+") as f:
        f.write(output_txt + '\n')
        f.close()

    h = int(training_time) // 3600
    m = (int(training_time) - h*3600) // 60
    s = int(training_time) % 60

    # 将训练时长录入日志
    print("训练总时长:{}h{}m{}s".format(h, m, s))  # 训练总时长
    output_txt = "Training Time:%dh %dm %ds" % (h, m, s)
    with open(log_txt_save_path, "a+") as f:
        f.write(output_txt + '\n')
        f.close()

    # 清空使用过的gpu缓冲区
    torch.cuda.empty_cache()

    print("==================== Finish Model Training ====================")
    exit()
