
# 项目介绍
使用哨兵一号遥感影像差分干涉相位图进行采矿沉陷分割的改进Unet网络模型

# 环境依赖
numpy、pytorch、tqdm、thop、open-cv、matplotlib

# 目录结构描述
    ├── README.txt           // 帮助文档
    ├── main.py                       // 主函数
    ├── train.py                       // 训练及验证函数
    ├── predict.py                   // 测试函数
    ├── MODEL                      // 模型文件夹
           ├───── UNet.py                         // Unet模型
           ├───── EMAGUNetPP.py           // EMAGUnet++模型
    ├── utils                           // 工具文件夹
            ├───── dataset.py                    // 数据加载工具函数
           ├───── draw.py                        // 可视化工具函数
           ├───── score.py                       // 评估指标工具函数
           ├───── seed.py                        // 随机种子工具函数
    ├── logs                      // 日志保存文件夹
    ├── predict_result        // 测试结果保存文件夹
    ├── data                      // 数据集文件夹
           ├───── train                        // 训练数据
                    ├───── image                        // 影像
                    ├───── label                          // 标签
           ├───── eval                         // 验证数据
                    ├───── image                        // 影像
                    ├───── label                          // 标签
           ├───── test                         // 测试数据
                    ├───── image                        // 影像
                    ├───── label                          // 标签


引用：Yinke Zhu, Tianhua Chen, Jinghui Fan, Hongli Zhao, Jiahui Lin, Guang Liu, and Shibiao Bai "Segmentation of mining subsidence areas in D-InSAR interferometric phase images using improved UNet++ network," Journal of Applied Remote Sensing 18(3), 034522 (27 September 2024). https://doi.org/10.1117/1.JRS.18.034522