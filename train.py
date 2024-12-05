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

    # train_data = TrainLoader(train_data_path)   
    # eval_data = EvalLoader(eval_data_path)  

    train_data = TrainLoader_new(train_data_path)  
    eval_data = EvalLoader_new(eval_data_path)

    print("Epoch:{} batch_size:{}  Ori_LearnRate：{}".format(train_epochs, train_batch_size, train_lr))

    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_data, batch_size=train_batch_size, shuffle=True)

    # Definition optimizer
    optimizer = optim.Adam(net.parameters(), lr=train_lr, betas=(0.9, 0.99))  # Adam
    # optimizer = optim.SGD(net.parameters(),lr=train_lr,momentum=0.8)  # Momentum
    # optimizer = optim.RMSprop(net.parameters(), lr=train_lr, alpha=0.9)    # RMSprop

    # Define a learning rate decay strategy
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9, 15, 22], gamma=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

    # Defined loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_iou = 0
    early_stop_counter = 0
    training_time = 0

    train_data_size = len(train_data)
    eval_data_size = len(eval_data)

    print("TrainDataSize：{}   EvalDataSize:{}".format(train_data_size, eval_data_size))

    output_txt = "EPOCH: Train_Loss: Eval_Loss: Eval_iou: Eval_PA:"
    with open(log_txt_save_path, "a+") as f:
        f.truncate(0)
        f.write(output_txt + '\n')
        f.close()

    for train_epoch in range(train_epochs):

        print("===================== EPOCH {}/{} =====================".format((train_epoch+1), train_epochs))

        net.train()
        total_train_loss = 0 
        total_train_iou = 0
        total_train_PA = 0 
        start = time.time()
      
        print("Traning")
        train_pbar = tqdm(total=train_data_size)
        for image, label in train_dataloader: 

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = net(image)
            loss = loss_fn(output, label)
            iou = iou_score(output, label)
            PA = pixel_accuracy(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss = total_train_loss + loss.item()
            total_train_iou = total_train_iou + iou
            total_train_PA = total_train_PA + PA

            time.sleep(0.05)
            train_pbar.update(1 * train_batch_size)
        train_pbar.close()

        end = time.time()
        time_elapsed = end - start
        training_time = training_time + time_elapsed

        print("Train_Loss={}   Train_iou={}   Train_PA={}".format(
            ((total_train_loss * train_batch_size) / train_data_size),
            ((total_train_iou * train_batch_size) / train_data_size),
            ((total_train_PA * train_batch_size) / train_data_size)))

        net.eval()
        total_eval_loss = 0
        total_eval_iou = 0
        total_eval_PA = 0

        with torch.no_grad():
            print("Eval")
            for image, label in eval_dataloader:

                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                output = net(image)
                loss = loss_fn(output, label)
                iou = iou_score(output, label)
                PA = pixel_accuracy(output, label)

                total_eval_loss = total_eval_loss + loss.item()
                total_eval_iou = total_eval_iou + iou 
                total_eval_PA = total_eval_PA + PA

        print("Eval_Loss={}   Eval_iou={}   Eval_PA={}".format(
            ((total_eval_loss * train_batch_size) / eval_data_size),
            ((total_eval_iou * train_batch_size) / eval_data_size),
            ((total_eval_PA * train_batch_size) / eval_data_size)))

        scheduler.step(train_epoch)

        output_txt = "%d %f %f %f %f" % ((train_epoch + 1),
                                      ((total_train_loss * train_batch_size) / train_data_size),
                                      ((total_eval_loss * train_batch_size) / eval_data_size),
                                      ((total_eval_iou * train_batch_size) / eval_data_size),
                                      ((total_eval_PA * train_batch_size) / eval_data_size))
        with open(log_txt_save_path, "a+") as f:
            f.write(output_txt + '\n')
            f.close()

        if total_eval_iou > best_iou:
            best_iou = total_eval_iou
        # if total_eval_loss < best_loss:
        #     best_loss = total_eval_loss
            torch.save(net.state_dict(), log_pth_save_path)
            print("========【EPOCH {} Already refreshed】========".format(train_epoch+1))
            early_stop_counter = 0
        else:
            print("========【EPOCH {} Not refreshed】========".format(train_epoch + 1))
            early_stop_counter = early_stop_counter + 1

        if early_stop_counter >= early_stop_number:
            print("==================== Early Stopping ====================")
            break

    draw(log_txt_save_path, loss_picture_save_path)

    output_txt = "Epoch:%d batchsize:%d Ori_LearnRate:%f" % (train_epochs, train_batch_size, train_lr)
    with open(log_txt_save_path, "a+") as f:
        f.write(output_txt + '\n')
        f.close()

    h = int(training_time) // 3600
    m = (int(training_time) - h*3600) // 60
    s = int(training_time) % 60

    print("TrainTime:{}h{}m{}s".format(h, m, s))
    output_txt = "Training Time:%dh %dm %ds" % (h, m, s)
    with open(log_txt_save_path, "a+") as f:
        f.write(output_txt + '\n')
        f.close()

    torch.cuda.empty_cache()

    print("==================== Finish Model Training ====================")
    exit()
