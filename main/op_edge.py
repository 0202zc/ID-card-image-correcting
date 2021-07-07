from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import *
import cv2
import math
import numpy as np


# 直方图均值化
def img_equalize(src):
    img1 = cv2.imread(src, 0)
    equ = cv2.equalizeHist(img1)
    cv2.imwrite("equ.png", equ)


# 根据四个角点进行透视变换
def outline_perspective_transform(img, upper_left, upper_right, bottom_left, bottom_right):
    width = 1180
    height = 774

    # 原图中的四个角点
    pts1 = np.float32([[upper_left], [upper_right], [bottom_left], [bottom_right]])
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(img, M, (width, height))

    return dst


def boundary(img_file):
    image = cv2.imread(img_file, 1)

    height = image.shape[0]
    width = image.shape[1]

    dst = np.zeros((height, width, 1), np.uint8)

    return dst


# 平面剪裁（不进行透视）
def img_cropped(image, x_1, y_1, x_2, y_2):
    if (x_1 < 0 or x_1 >= image.shape[1]) and (y_1 < 0 or y_1 >= image.shape[0]):
        print("error parameters")
        return Exception
    if (x_2 < 0 or x_2 >= image.shape[1]) and (y_2 < 0 or y_2 >= image.shape[0]):
        print("error parameters")
        return Exception

    cropped = image[y_1:y_2, x_1:x_2]

    return cropped


# 画边界线
def draw_lines(img, re_img):
    img1 = img.copy()
    img = cv2.GaussianBlur(img, (9, 9), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value %s" % ret)

    ###############################滤波##################################
    # 均值滤波
    img_mean = cv2.blur(binary, (5, 5))

    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(binary, (5, 5), 0)

    # 中值滤波
    img_median = cv2.medianBlur(binary, 5)

    # 双边滤波
    img_bilater = cv2.bilateralFilter(binary, 9, 75, 75)
    #####################################################################

    cv2.imwrite('./pretreatment.png', img_median)
    edges = cv2.Canny(img_median, 50, 150, apertureSize=3)

    #####################################################################
    tmp = np.zeros(img.shape, np.uint8)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 10000:
            res = cv2.drawContours(tmp, cnt, -1, (255, 255, 255), 1)
            cv2.imwrite('cnt.png', res)
            break
    #####################################################################

    tmp = cv2.imread("cnt.png", 0)
    threshold = 0
    lines = cv2.HoughLines(tmp, 1, np.pi / 180, 140)
    while len(lines) is not 8:
        threshold += 1
        lines = cv2.HoughLines(tmp, 1, np.pi / 180, threshold)

    lines_array = []
    theta_array = []
    tmp_line_array = []

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        this_theta = theta * 180 / np.pi
        print("the theta is: %s" % str(this_theta))

        flag = False
        theta_array.append(this_theta)
        tmp_line_array.append([x1, y1, x2, y2])
        this_line = [x1, y1, x2, y2]

        # 筛选边缘直线
        for i in range(0, len(theta_array) - 1):
            cross = intersection_1(this_line, tmp_line_array[i])
            if 0 < abs(theta_array[i] - this_theta) < 5:
                if len(cross) != 0 and cross[0] < 1500 and cross[1] < 1500:
                    flag = True
                    break
            elif abs(theta_array[i] - this_theta) == 0:
                if get_point_line_distance([this_line[0], this_line[1]], tmp_line_array[i]) < 3:
                    flag = True
                    break

        if not flag:
            cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 线段延长处理
            if 0 < x1 < 1000 and 0 < y1 < 1000:
                x1 = 2 * x1 - x2
                y1 = 2 * y1 - y2
            if 0 < x2 < 1000 and 0 < y2 < 1000:
                x2 = 2 * x2 - x1
                y2 = 2 * y2 - y1

            lines_array.append([x1, y1, x2, y2])

    print(lines_array)
    cv2.imwrite("./houghlines.png", img1)

    return lines_array


def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0]
    line_s_y = line[1]
    line_e_x = line[2]
    line_e_y = line[3]
    # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)

    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    b = line_s_y - k * line_s_x

    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis


# 判断 (xk, yk) 是否在「线段」(x1, y1)~(x2, y2) 上
# 这里的前提是 (xk, yk) 一定在「直线」(x1, y1)~(x2, y2) 上
def inside(x1, y1, x2, y2, xk, yk):
    # 若与 x 轴平行，只需要判断 x 的部分
    # 若与 y 轴平行，只需要判断 y 的部分
    # 若为普通线段，则都要判断
    return (x1 == x2 or min(x1, x2) <= xk <= max(x1, x2)) and (y1 == y2 or min(y1, y2) <= yk <= max(y1, y2))


def update(ans, xk, yk):
    # 将一个交点与当前 ans 中的结果进行比较
    # 若更优则替换
    return [xk, yk] if not ans or [xk, yk] < ans else ans


def intersection(line1, line2):
    # 取四点坐标
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    ans = list()
    # 判断 (x1, y1)~(x2, y2) 和 (x3, y3)~(x4, y3) 是否平行
    if (y4 - y3) * (x2 - x1) == (y2 - y1) * (x4 - x3):
        # 若平行，则判断 (x3, y3) 是否在「直线」(x1, y1)~(x2, y2) 上
        if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
            # 判断 (x3, y3) 是否在「线段」(x1, y1)~(x2, y2) 上
            if inside(x1, y1, x2, y2, x3, y3):
                ans = update(ans, x3, y3)
            # 判断 (x4, y4) 是否在「线段」(x1, y1)~(x2, y2) 上
            if inside(x1, y1, x2, y2, x4, y4):
                ans = update(ans, x4, y4)
            # 判断 (x1, y1) 是否在「线段」(x3, y3)~(x4, y4) 上
            if inside(x3, y3, x4, y4, x1, y1):
                ans = update(ans, x1, y1)
            # 判断 (x2, y2) 是否在「线段」(x3, y3)~(x4, y4) 上
            if inside(x3, y3, x4, y4, x2, y2):
                ans = update(ans, x2, y2)
        # 在平行时，其余的所有情况都不会有交点
    else:
        # 联立方程得到 t1 和 t2 的值
        t1 = (x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / (
                (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1))
        t2 = (x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / (
                (x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3))
        # 判断 t1 和 t2 是否均在 [0, 1] 之间
        if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
            ans = [x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1)]

    return ans


def intersection_1(line1, line2):
    # 取四点坐标
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    det = lambda a, b, c, d: a * d - b * c
    d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
    p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
    q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
    if d != 0:
        lam, eta = p / d, q / d
        if not (0 <= lam <= 1 and 0 <= eta <= 1): return []
        return [lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2]
    if p != 0 or q != 0: return []
    t1, t2 = sorted([[line1[0], line1[1]], [line1[2], line1[3]]]), sorted([[line2[0], line2[1]], [line2[2], line2[3]]])
    if t1[1] < t2[0] or t2[1] < t1[0]: return []
    return max(t1[0], t2[0])


# 整理直线交点信息
def cross_point_list(array):
    cross_list = []

    for i in range(0, 3):
        for j in range(i + 1, 4):

            temp = intersection_1(array[i], array[j])
            if temp is None or len(temp) == 0 or abs(temp[0]) > 2000 or abs(temp[1]) > 2000:
                continue
            else:
                if len(cross_list) < 4:
                    print(temp)
                    cross_list.append(temp)
    return cross_list


# 根据长短边识别角点的相对位置【前提：list的第一个元素为左上角角点】
def handle_edge(vertex_list):
    x1 = vertex_list[0][0]
    y1 = vertex_list[0][1]
    distances = []
    return_list = [vertex_list[0]]
    for i in range(1, 4):
        temp_x = vertex_list[i][0]
        temp_y = vertex_list[i][1]
        d = np.sqrt((temp_x - x1) * (temp_x - x1) + (temp_y - y1) * (temp_y - y1)) / 100
        distances.append([d, i])

    distances.sort()
    bottom_left = vertex_list[distances[0][1]]
    upper_right = vertex_list[distances[1][1]]
    bottom_right = vertex_list[distances[2][1]]

    return_list = get_the_A([[x1, y1], bottom_left, upper_right, bottom_right])

    return return_list


def get_the_A(vertex_list):
    # 前提：身份证【正方向】底部长边与照片边缘底部的夹角小于π/2
    # vertex_list: A, B, D, C，其中A为待测点，B为A的短边相邻顶点，D为A的长边相邻顶点，C为A的对角顶点
    # 事先已对list进行相对A的距离排序
    # A = vertex_list[0], B = vertex_list[1], D = vertex_list[2], C = vertex_list[3]
    short_edge = [vertex_list[0], vertex_list[1]]
    long_edge = [vertex_list[2], vertex_list[3]]

    x_A = short_edge[0][0]
    y_A = short_edge[0][1]
    x_B = short_edge[1][0]
    y_B = short_edge[1][1]
    x_D = long_edge[0][0]
    y_D = long_edge[0][1]
    x_C = long_edge[1][0]
    y_C = long_edge[1][1]

    if y_A > y_B:  # A在B下方
        if x_A > x_D:  # A在D右边
            return [[x_C, y_C], [x_B, y_B], [x_D, y_D], [x_A, y_A]]
        else:  # A在D左边
            return [[x_B, y_B], [x_C, y_C], [x_A, y_A], [x_D, y_D]]
    else:  # A在B上方
        if x_A > x_D:  # A在D右边
            return [[x_D, y_D], [x_A, y_A], [x_C, y_C], [x_B, y_B]]
        else:  # A在D左边
            return [[x_A, y_A], [x_D, y_D], [x_B, y_B], [x_C, y_C]]


# 方法汇总函数
def handle_point_list(vertex_list):
    temp_list = handle_edge(vertex_list)
    return temp_list


def img_rotate(src, dst):
    img = cv2.imread(src)

    # 获取图像的尺寸
    # 旋转中心
    (h, w) = img.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    rotated = cv2.warpAffine(img, M, (nW, nH))
    cv2.imwrite(dst, rotated)


# 边界填充
def fill_edge(src):
    img = cv2.imread(src)

    height = img.shape[0]
    width = img.shape[1]
    fill_up = []
    fill_down = []
    fill_left = []
    fill_right = []

    for i in range(width):
        for j in range(height):
            if abs(width - 1 - i) <= 30 or abs(height - 1 - j) <= 30 or i <= 30 or j <= 30:
                if (img[j][i][0] < 220 and img[j][i][1] < 220 and img[j][i][2] < 220) or (
                        img[j][i][0] > 239 and img[j][i][1] > 239 and img[j][i][2] > 239):
                    if 30 <= j <= height - 1 - 30 and i <= 30:
                        fill_left.append([j, i])
                    elif 30 <= j <= height - 1 - 30 and width - 1 - 30 <= i <= width - 1:
                        fill_right.append([j, i])
                    elif j <= 30:
                        fill_up.append([j, i])
                    elif height - 1 - 30 <= j <= height - 1:
                        fill_down.append([j, i])

    for left in fill_left:
        img[left[0]][left[1]] = img[left[0]][left[1] + 5]
    for right in fill_right:
        img[right[0]][right[1]] = img[right[0]][right[1] - 5]
    for up in fill_up:
        if up[1] > img.shape[1] / 2:
            img[up[0]][up[1]] = img[up[0] + 30][up[1] - 30]
        elif up[1] <= img.shape[1] / 2:
            img[up[0]][up[1]] = img[up[0] + 30][up[1] + 30]
    for down in fill_down:
        if down[1] > img.shape[1] / 2:
            img[down[0]][down[1]] = img[down[0] - 31][down[1] - 30]
        elif down[1] <= img.shape[1] / 2:
            img[down[0]][down[1]] = img[down[0] - 31][down[1] + 30]
    cv2.imwrite("./result/IDCard_filled.png", img)
