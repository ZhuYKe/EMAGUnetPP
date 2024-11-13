#!/usr/bin/env python
# coding=utf-8
from matplotlib import pyplot as plt

def draw(path, loss_picture_save_path):
    with open(path) as file_object:
        lines = file_object.readlines()

    file1 = []
    for line in lines:
        row = line.split()
        file1.append(row)

    epoch = []
    train_loss = []
    eval_loss = []

    for row1 in file1:
        epoch.append(row1[0])
        train_loss.append(row1[1])
        eval_loss.append(row1[2])

    del epoch[0]
    del train_loss[0]
    del eval_loss[0]

    int_epoch = []
    for str_Epoch in epoch:
        a = int(float(str_Epoch))
        int_epoch.append(a)

    float_train_loss = []
    for str_T in train_loss:
        a = float(str_T)
        float_train_loss.append(a)

    float_eval_loss = []
    for str_Eval in eval_loss:
        a = float(str_Eval)
        float_eval_loss.append(a)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    ax1.plot(int_epoch, float_train_loss, c="red", label='Train_Loss')
    ax1.plot(int_epoch, float_eval_loss, c="blue", label='Eval_Loss')
    plt.legend()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    ax1.set_title("Train_Loss and Eval_Loss", fontsize=18)
    # ax1.tick_params(direction='in')

    plt.savefig(loss_picture_save_path)
    plt.show()


if __name__ == "__main__":
    Log_save_txt = "D:/ZYK/Unet/logs/chenxianmoni/Unet/logs.txt"  # 日志文件保存路径
    Loss_picture_save_path = "D:/ZYK/Unet/figure.png"
    draw(path=Log_save_txt, loss_picture_save_path=Loss_picture_save_path)
