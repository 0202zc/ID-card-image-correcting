import cv2
import numpy as np
import math
import time


# 根据四个角点进行透视变换
def outline_perspective_transform(img, left_up, right_up, left_down, right_down):
    width = 1180
    height = 774

    # dst = np.zeros((height, width, 3), np.uint8)
    # # for i in range(0, height - 1):
    # #     for j in range(0, width - 1):
    # #         dst[i, j] = 0
    # # rows, cols = img.shape[:2]

    # 原图中的四个角点
    pts1 = np.float32([[left_up], [right_up], [left_down], [right_down]])
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
def draw_lines(img):
    img1 = img.copy()
    img2 = img.copy()
    img = cv2.GaussianBlur(img, (9, 9), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    threshold = 0
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)
    while lines.size != 8:
        threshold += 1
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    lines_array = []

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

        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        lines_array.append([x1, y1, x2, y2])

    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 131, 300, 5)
    #
    # for line in lines:
    #     x1 = line[0][0]
    #     y1 = line[0][1]
    #     x2 = line[0][2]
    #     y2 = line[0][3]
    #     cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('houghlines3', img1)
    # cv2.imshow('edges', img2)
    cv2.waitKey(0)
    cv2.imwrite("houghlines.png", img1)
    cv2.destroyAllWindows()
    return lines_array


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if x2 - x1 == 0:
        return None
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        if k1 == k2:  # 两条平行线
            return None
        else:
            x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


# 整理直线交点信息
def cross_point_list(array):
    cross_list = []

    for i in range(0, 3):
        for j in range(1, 4):
            if i >= j:
                continue
            else:
                temp = cross_point(array[i], array[j])
                if temp is None or abs(temp[0]) > 2000 or abs(temp[1]) > 2000:
                    continue
                else:
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
    left_down = vertex_list[distances[0][1]]
    right_up = vertex_list[distances[1][1]]
    right_down = vertex_list[distances[2][1]]

    return_list = get_the_A([[x1, y1], left_down, right_up, right_down])

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
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # 参数：原始图像 旋转参数 元素图像宽高
    rotated = cv2.warpAffine(img, M, (cols, rows))

    # # 显示图像
    # cv2.imshow("src", img)
    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(dst, rotated)

def fill_edge(src):
    src = "./result/IDCard.png"
    img = cv2.imread(src)

    height = img.shape[0]
    width = img.shape[1]
    fill_up = []
    fill_down = []
    fill_left = []
    fill_right = []

    for i in range(0, width - 1):
        for j in range(0, height - 1):
            if abs(width - 1 - i) < 30 or abs(height - 1 - j) < 30 or i < 30 or j < 30:
                if img[j][i][0] < 220 and img[j][i][1] < 220 and img[j][i][2] < 220:
                    if 30 <= j <= height - 1 - 30 and 0 <= i <= 30:
                        fill_left.append([j, i])
                        img[j][i] = (255, 0, 0)  # blue
                    elif 30 <= j <= height - 1 - 30 and width - 1 - 30 <= i <= width - 1:
                        fill_right.append([j, i])
                        img[j][i] = (0, 255, 0)  # green
                    elif 0 <= j <= 30:
                        fill_up.append([j, i])
                        img[j][i] = (0, 0, 255)  # red
                    elif height - 1 - 30 <= j <= height - 1:
                        fill_down.append([j, i])
                        img[j][i] = (255, 0, 255)  # purple
                    # img[j][i] = res[0]
                    # img[j][i + 1] = res[1]
                    # img[j][i + 2] = res[2]
                    # img[j + 1][i] = res[3]
                    # img[j + 1][i + 1] = res[4]
                    # img[j + 1][i + 2] = res[5]
                    # img[j + 2][i] = res[6]
                    # img[j + 2][i + 1] = res[7]
                    # img[j + 2][i + 2] = res[8]

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
    cv2.imshow("transformed", img)
    cv2.imwrite("./result/IDCard_filled.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    src = "../images/black_background/13.png"
    while True:
        img = cv2.imread(src)
        size = max(img.shape[0], img.shape[1]) / 1050
        img = cv2.resize(img, (int(img.shape[1] / size), int(img.shape[0] / size)), interpolation=cv2.INTER_AREA)

        try:
            array = draw_lines(img)
            cross_list = cross_point_list(array)
            result = handle_point_list(cross_list)
            img = outline_perspective_transform(img, result[0], result[1], result[2], result[3])
        except Exception as e:
            print(e)
            break

        # cv2.imshow("img", img)
        cv2.imwrite("./result/transformed.png", img)
        cv2.imshow("transformed", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        face_img("./result/transformed.png")
        result = face_test("./result/face.png")
        img_rotate("./result/transformed.png", "./result/transformed.png")

        result_trans = face_test("./result/transformed.png")
        if (not result) and (not result_trans):
            img_rotate(src, "./rotated.png")
            src = "./rotated.png"
        elif (not result) and result_trans:
            face_img("./result/transformed.png")
            break
        else:
            break
    cv2.imshow("face", cv2.imread("./result/face.png"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
