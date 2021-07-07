import cv2
import numpy as np


def face_img(src):
    face_width = 358
    face_height = 441

    image = cv2.imread(src)
    dst = np.zeros((face_height, face_width, 3), np.uint8)

    for i in range(135, 135 + face_height - 1):
        for j in range(730, 730 + face_width - 1):
            dst[i - 135, j - 730] = image[i, j]

    cv2.imwrite("./result/face.png", dst)
    # cv2.imshow('face', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


def face_test(impath):
    image = cv2.imread(impath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cade = cv2.CascadeClassifier(r'../face_recg/haarcascade_frontalface_default.xml')
    fa = face_cade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    if len(fa) == 0:
        return False
    else:
        for (x, y, w, h) in fa:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255.0), 2)
        cv2.imwrite('./result/cv_final.png', image)
        return True


def img_rotate(src, dst):
    img = cv2.imread(src)
    # 原图的高、宽 以及通道数
    rows, cols, channel = img.shape

    # 绕图像的中心旋转
    # 参数：旋转中心 旋转度数 scale
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    # 参数：原始图像 旋转参数 元素图像宽高
    rotated = cv2.warpAffine(img, M, (cols, rows))

    # # 显示图像
    # cv2.imshow("src", img)
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(dst, rotated)
