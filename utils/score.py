#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np


def iou_score(output, label):
    smooth = 0.000000000000001

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy
    if torch.is_tensor(label):
        label = label.data.cpu().numpy
    output_ = output() > 0.5
    label_ = label() > 0.5
    intersection = (output_ & label_).sum()
    union = (output_ | label_).sum()

    return intersection / (union + smooth)


def background_iou(output, label):
    smooth = 0.0000000001

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.data.cpu().numpy()

    output_ = output > 0.5
    label_ = label > 0.5
    output_ = ~output_
    label_ = ~label_

    intersection = np.sum(output_ & label_, axis=(1, 2))
    union = np.sum(output_ | label_, axis=(1, 2))
    iou = intersection / (union + smooth)

    return iou


def pixel_accuracy(output, label):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.data.cpu().numpy()

    output_ = output > 0.5
    label_ = label > 0.5

    correct_pixels = (output_ == label_).sum()
    total_pixels = output_.size
    accuracy = correct_pixels / total_pixels
    return accuracy


def PPV_score(output, label):
    smooth = 0.000000000000001

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.data.cpu().numpy()

    output_ = output > 0.5
    label_ = label > 0.5

    tp = np.logical_and(output_, label_).sum()
    fp = np.logical_and(output_, np.logical_not(label_)).sum()
    ppv = tp / (tp + fp + smooth)

    return ppv


def TPR_score(output, label):
    smooth = 0.000000000000001

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.data.cpu().numpy()

    output_ = output > 0.5
    label_ = label > 0.5

    tp = np.logical_and(output_, label_).sum()
    fn = np.logical_and(np.logical_not(output_), label_).sum()
    TPR = tp / (tp + fn + smooth)

    return TPR
