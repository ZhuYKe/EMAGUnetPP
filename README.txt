# Introduction
Improved Unet network model for mining subsidence segmentation using differential interferometric phase maps of Sentinel-1 remote sensing images

# requirements
numpy 1.12.6
pytorch 1.2.0
tqdm 4.66.1
thop 0.1.1
open-cv 4.8.1.78
matplotlib 3.5.3

# Directory structure description
    ├── README.txt                  // Help document
    ├── torch2onnx.py               // Translate torch model and pth to ONNX
    ├── onnxrun.py                  // Use onnx predict png
    ├── main.py                     // Principal function 
    ├── train.py                    // Train and Eval functions
    ├── predict.py                  // Test function
    ├── MODEL                       // Model folder
           ├───── UNet.py                    // Unet Model
           ├───── EMAGUNetPP.py              // EMAGUnet++ Model
    ├── utils                       // Tools folder
            ├───── dataset.py                // Data loading tool function
           ├───── draw.py                    // Visual tool function
           ├───── score.py                   // Evaluate indicator tool functions
           ├───── seed.py                    // Random seed tool function
    ├── logs                       // Log folder
    ├── predict_result             // Test Result folder
    ├── data                       // Dataset folder
           ├───── train                     // Train Dataset
                    ├───── image                      // image
                    ├───── label                      // label
           ├───── eval                      // Eval Dataset
                    ├───── image                      // image
                    ├───── label                      // label
           ├───── test                      // Test Dataset
                    ├───── image                      // image
                    ├───── label                      // label


# Cite：
Yinke Zhu, Tianhua Chen, Jinghui Fan, Hongli Zhao, Jiahui Lin, Guang Liu, and Shibiao Bai "Segmentation of mining subsidence areas in D-InSAR interferometric phase images using improved UNet++ network," Journal of Applied Remote Sensing 18(3), 034522. https://doi.org/10.1117/1.JRS.18.034522
