# import tensorflow as tf
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import tensorflow as tf
import numpy as np
import cv2

image = cv2.imread('./images/1.png')  # 图像读取
size = 2
image = cv2.resize(image, (int(image.shape[1] / size), int(image.shape[0] / size)), interpolation=cv2.INTER_AREA)

gray = tf.cast(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype="float32")  # 转灰度图并转换为tensor型

image_x = tf.reshape(gray, [1, 222, 395, 1])  # 变换维度（batch，height，weight，channel）

kernal = tf.constant([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype="float32")  # sobel算子

input_kernal = tf.reshape(kernal, [3, 3, 1, 1])  # [weight, height, channel, out_channel]

conv = tf.nn.conv2d(image_x, input_kernal, strides=[1, 1, 1, 1], padding='VALID')  # 卷积

with tf.Session() as sess:  # 建图
    y = sess.run(conv)  # run
    y_1 = sess.run(tf.transpose(y, [3, 0, 1, 2]))  # 反卷积，即维度转置
    cv2.imshow("sobel", y_1[0][0])
