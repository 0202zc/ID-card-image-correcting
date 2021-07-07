# ID-card-image-correcting
### 介绍

- 基于 OpenCV 的身份证图像矫正软件
- 给一个简单背景下的含有身份证的照片，对其所产生的几何畸变进行**矫正**，使处理后的照片几何尺寸的比例关系接近于真实身份证的几何尺寸比例关系。
- 本软件为本人的毕业设计，使用 Python 3 开发，对深色背景的身份证照片有较好的矫正效果
- 参考 CSDN 博文：[如何判断身份证顶点的相对位置](https://blog.csdn.net/kkm09/article/details/104938358)

### Installation

#### Requirement

- Python 3.6
- PyQt5 5.15.4
- OpenCV 4.3、opencv_python 4.3
- Numpy 1.14.3

 #### Step

1. clone本项目到本地
2. 安装环境，在 `ID-card-image-correcting\main`文件夹下执行`pip install -r requirements.txt`
3. 运行 `python main_window.py`

