# Readme

### 介绍

本文件夹为Python项目的主文件夹：

- `main_window.py` 主程序文件，包含前端窗口和函数功能调用
- `op_edge.py ` 边缘处理文件，包括图像的降噪处理和身份证边缘的识别
- `op_face.py` 人脸识别文件
- `recog.py` 、`recog.ui` 前端设计文件，基于PyQt5
- `recognition.py` 图像变换处理文件，具体介绍参照博文：[《如何判断身份证顶点的相对位置》](https://blog.csdn.net/kkm09/article/details/10493835)
- `result` 文件夹存放矫正后的图像

### 运行

- 先安装依赖：`pip install -r requirements.txt`
  - OpenCV可能需要单独安装
- 运行 main_window.py，显示程序主窗体
- 点击“选择图像”，最好控制图像分辨率在 `1129 * 666` 左右或图像大小在2MB以下，且选择深色背景的图像
- 点击“图像矫正”可执行程序步骤
- 执行完毕后，点击后续的按钮可查看矫正过程的效果

### 注意

本软件矫正精度与前期拍摄效果相关