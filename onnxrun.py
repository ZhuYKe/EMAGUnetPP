#!/usr/bin/env python
# coding=utf-8
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import cv2


def model_predict(input_png, output_png, model_oonx):
    test_data = cv2.imread(input_png)
    test_data = cv2.cvtColor(test_data, cv2.COLOR_BGR2GRAY)
    test_data = test_data.reshape(1, 1, test_data.shape[0], test_data.shape[1])

    session = ort.InferenceSession(model_oonx)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image = test_data.astype(np.float32)

    result = session.run([output_name], {input_name: image})
    output = result[0]
    output = output[0, 0]  # 提取结果
    output[output >= 0.5] = 255  # 处理结果
    output[output < 0.5] = 0
    cv2.imwrite(output_png, output)  # 保存图片

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("请按格式输入")
        print("<script_name> <input_png> <oonx_model_path> <output_png>")
        print("input_png：待处理的png切片（256*256）")
        print("oonx_model_path：ONNX格式已预训练好的神经网络模型")
        print("output_png：采矿沉陷分割结果png文件")
        sys.exit(1)

    input_png = sys.argv[1]
    oonx_model_path = sys.argv[2]
    output_png = sys.argv[3]
    # model_oonx = "./UNet_Segmentation_pth=20040222_1231.onnx"
    # input_png = "input.png"
    # output_png = "output.png"

    model_predict(input_png, output_png, oonx_model_path)


