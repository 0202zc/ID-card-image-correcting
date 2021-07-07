from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import *
import cv2
import sys
import time

import op_edge as op_eg
import op_face as op_fc


def do_action(src):
    i = 0
    while i <= 3:
        img = cv2.imread(src)
        size = max(img.shape[0], img.shape[1]) / 1050
        img = cv2.resize(img, (int(img.shape[1] / size), int(img.shape[0] / size)), interpolation=cv2.INTER_AREA)
        cv2.imwrite("resize.png", img)

        try:
            op_eg.img_equalize("resize.png")
            array = op_eg.draw_lines(img, cv2.imread("equ.png"))
            cross_list = op_eg.cross_point_list(array)
            result = op_eg.handle_point_list(cross_list)
            img = op_eg.outline_perspective_transform(img, result[0], result[1], result[2], result[3])
        except Exception as e:
            print(e)
            return e

        # cv2.imshow("img", img)
        cv2.imwrite("./result/transformed.png", img)
        cv2.imwrite("./result/IDCard.png", img)

        op_eg.fill_edge("./result/IDCard.png")

        # cv2.imshow("transformed", img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        op_fc.face_img("./result/transformed.png")
        result = op_fc.face_test("./result/face.png")
        op_fc.img_rotate("./result/transformed.png", "./result/transformed.png")

        result_trans = op_fc.face_test("./result/transformed.png")
        if (not result) and (not result_trans):
            op_eg.img_rotate(src, "./rotated.png")
            src = "./rotated.png"
            i += 1
        elif (not result) and result_trans:
            op_fc.img_rotate("./result/IDCard.png", "./result/IDCard.png")
            op_fc.img_rotate("./result/IDCard_filled.png", "./result/IDCard_filled.png")
            op_fc.face_img("./result/transformed.png")
            op_eg.fill_edge("./result/IDCard.png")
            break
        else:
            break
    # cv2.imshow("face", cv2.imread("./result/face.png"))
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    if i == 4:
        return False
    return True


class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(824, 493)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        MainWindow.setFixedSize(self.width(), self.height())

        self.view_IDCard = QtWidgets.QGraphicsView(MainWindow)
        self.view_IDCard.setGeometry(QtCore.QRect(220, 90, 590, 387))
        self.view_IDCard.setObjectName("view_IDCard")
        self.horizontalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 791, 71))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setDisabled(True)
        self.horizontalLayout.addWidget(self.lineEdit)
        self.btn_input = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_input.setObjectName("btn_input")
        self.horizontalLayout.addWidget(self.btn_input)
        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 90, 181, 161))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_output = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_output.setObjectName("btn_output")
        self.verticalLayout.addWidget(self.btn_output)
        self.btn_pre = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_pre.setObjectName("btn_pre")
        self.verticalLayout.addWidget(self.btn_pre)
        self.btn_edge = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_edge.setObjectName("btn_edge")
        self.verticalLayout.addWidget(self.btn_edge)
        self.btn_trans = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_trans.setObjectName("btn_trans")
        self.verticalLayout.addWidget(self.btn_trans)
        self.btn_fill = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_fill.setObjectName("btn_fill")
        self.verticalLayout.addWidget(self.btn_fill)
        self.view_face = QtWidgets.QGraphicsView(MainWindow)
        self.view_face.setGeometry(QtCore.QRect(20, 260, 179, 221))
        self.view_face.setObjectName("view_face")
        self.label_2 = QtWidgets.QLabel(MainWindow)
        self.label_2.setGeometry(QtCore.QRect(220, 90, 590, 387))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(MainWindow)
        self.label_3.setGeometry(QtCore.QRect(20, 260, 179, 221))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")

        self.btn_pre.setDisabled(True)
        self.btn_edge.setDisabled(True)
        self.btn_trans.setDisabled(True)
        self.btn_fill.setDisabled(True)
        self.btn_output.setDisabled(True)

        self.retranslateUi(MainWindow)

        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.btn_input.clicked.connect(self.openfile)
        self.btn_output.clicked.connect(self.do_recognition)
        self.btn_pre.clicked.connect(self.do_pretreatment)
        self.btn_edge.clicked.connect(self.do_edge)
        self.btn_trans.clicked.connect(self.do_transform)
        self.btn_fill.clicked.connect(self.do_fill)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "身份证照片矫正"))
        MainWindow.setWindowTitle(_translate("MainWindow", "身份证照片矫正"))
        MainWindow.setWindowIcon(QIcon("../images/身份证带圈.png"))
        self.label.setText(_translate("MainWindow", "图像路径："))
        self.btn_input.setText(_translate("MainWindow", "选择图片"))
        self.btn_output.setText(_translate("MainWindow", "图像矫正"))
        self.btn_pre.setText(_translate("MainWindow", "图像预处理"))
        self.btn_edge.setText(_translate("MainWindow", "边缘提取"))
        self.btn_trans.setText(_translate("MainWindow", "透视变换"))
        self.btn_fill.setText(_translate("MainWindow", "边缘填充"))

    def openfile(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '../images', 'PNG Files(*.png);;JPEG Files(*.jpg)')[0]
        self.lineEdit.setText(openfile_name)

        img_IDCard = cv2.imread(openfile_name)

        jpg = QtGui.QPixmap(openfile_name).scaled(self.label_2.width(), self.label_2.height())
        print(self.lineEdit.text())

        if self.lineEdit.text() != '':
            self.btn_output.setDisabled(False)
            self.btn_pre.setDisabled(True)
            self.btn_edge.setDisabled(True)
            self.btn_trans.setDisabled(True)
            self.btn_fill.setDisabled(True)
        else:
            self.btn_output.setDisabled(True)
            self.btn_pre.setDisabled(True)
            self.btn_edge.setDisabled(True)
            self.btn_trans.setDisabled(True)
            self.btn_fill.setDisabled(True)

        self.label_2.setPixmap(jpg)
        self.label_2.setScaledContents(True)
        # 清空面板
        jpg = QtGui.QPixmap("null").scaled(self.label_2.width(), self.label_2.height())
        self.label_3.setPixmap(jpg)

    # import image_recognition.final_program.recognition
    def do_recognition(self):
        openfile_name = self.lineEdit.text()

        if openfile_name == "" or not openfile_name:
            reply = QMessageBox.question(self, '提示', '请选择文件。',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == 16384:
                self.openfile()
                if self.lineEdit.text() != '':
                    self.btn_output.setDisabled(False)
                    self.btn_pre.setDisabled(True)
                    self.btn_edge.setDisabled(True)
                    self.btn_trans.setDisabled(True)
                    self.btn_fill.setDisabled(True)
                else:
                    self.btn_output.setDisabled(True)
                    self.btn_pre.setDisabled(True)
                    self.btn_edge.setDisabled(True)
                    self.btn_trans.setDisabled(True)
                    self.btn_fill.setDisabled(True)
            else:
                return
        if openfile_name == "" or not openfile_name:
            return

        # 清空面板
        jpg = QtGui.QPixmap("null").scaled(self.label_2.width(), self.label_2.height())
        self.label_3.setPixmap(jpg)

        start = time.clock()
        flag = do_action(openfile_name)
        end = time.clock()

        print("running time is: %s" % (end - start))

        if flag == True:
            img_face = cv2.imread("./result/face.png")
            jpg = QtGui.QPixmap("./result/face.png").scaled(self.label_3.width(), self.label_3.height())
            self.label_3.setPixmap(jpg)

            jpg = QtGui.QPixmap("./result/IDCard_filled.png").scaled(self.label_2.width(), self.label_2.height())
            self.label_2.setPixmap(jpg)

            QMessageBox.question(self, 'Message',
                                 '提取成功！文件保存至' + 'image_recognition/final_program/result/face.png',
                                 QMessageBox.Yes)
            self.btn_pre.setDisabled(False)
            self.btn_edge.setDisabled(False)
            self.btn_trans.setDisabled(False)
            self.btn_fill.setDisabled(False)
            self.btn_output.setDisabled(False)
        else:
            print(flag)
            QMessageBox.question(self, 'Error', str(flag), QMessageBox.Yes)
            reply = QMessageBox.question(self, '提示', '角度刁钻，请选择其他文件。',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == 16384:
                self.openfile()
                if self.lineEdit.text() != '':
                    self.btn_output.setDisabled(False)
                    self.btn_pre.setDisabled(True)
                    self.btn_edge.setDisabled(True)
                    self.btn_trans.setDisabled(True)
                    self.btn_fill.setDisabled(True)
                else:
                    self.btn_output.setDisabled(True)
                    self.btn_pre.setDisabled(True)
                    self.btn_edge.setDisabled(True)
                    self.btn_trans.setDisabled(True)
                    self.btn_fill.setDisabled(True)

    def do_pretreatment(self):
        jpg = QtGui.QPixmap("./pretreatment.png").scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)
        jpg = QtGui.QPixmap("").scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)

    def do_edge(self):
        jpg = QtGui.QPixmap("./cnt.png").scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)
        jpg = QtGui.QPixmap("").scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)

    def do_transform(self):
        jpg = QtGui.QPixmap("./result/IDCard.png").scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)
        jpg = QtGui.QPixmap("./result/face.png").scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)

    def do_fill(self):
        jpg = QtGui.QPixmap("./result/IDCard_filled.png").scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)
        jpg = QtGui.QPixmap("./result/face.png").scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)

    # def setupUi(self, MainWindow):
    #     MainWindow.setObjectName("MainWindow")
    #     MainWindow.resize(829, 415)
    #     self.centralWidget = QtWidgets.QWidget(MainWindow)
    #     self.centralWidget.setObjectName("centralWidget")
    #     MainWindow.setFixedSize(self.width(), self.height())
    #
    #     self.view_IDCard = QtWidgets.QGraphicsView(MainWindow)
    #     self.view_IDCard.setGeometry(QtCore.QRect(220, 10, 590, 387))
    #     self.view_IDCard.setObjectName("view_IDCard")
    #     self.horizontalLayoutWidget = QtWidgets.QWidget(MainWindow)
    #     self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 181, 80))
    #     self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
    #     self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
    #     self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
    #     self.horizontalLayout.setObjectName("horizontalLayout")
    #     self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
    #     self.label.setObjectName("label")
    #     self.horizontalLayout.addWidget(self.label)
    #     self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
    #     self.lineEdit.setObjectName("lineEdit")
    #     self.lineEdit.setDisabled(True)
    #     self.horizontalLayout.addWidget(self.lineEdit)
    #     self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
    #     self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 90, 181, 80))
    #     self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
    #     self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
    #     self.verticalLayout.setContentsMargins(0, 0, 0, 0)
    #     self.verticalLayout.setObjectName("verticalLayout")
    #     self.btn_input = QtWidgets.QPushButton(self.verticalLayoutWidget)
    #     self.btn_input.setObjectName("btn_input")
    #     self.verticalLayout.addWidget(self.btn_input)
    #     self.btn_output = QtWidgets.QPushButton(self.verticalLayoutWidget)
    #     self.btn_output.setObjectName("btn_output")
    #     self.verticalLayout.addWidget(self.btn_output)
    #     self.view_face = QtWidgets.QGraphicsView(MainWindow)
    #     self.view_face.setGeometry(QtCore.QRect(20, 180, 179, 221))
    #     self.view_face.setObjectName("view_face")
    #     self.label_2 = QtWidgets.QLabel(MainWindow)
    #     self.label_2.setGeometry(QtCore.QRect(220, 10, 590, 387))
    #     self.label_2.setText("")
    #     self.label_2.setObjectName("label_IDCard")
    #     self.label_3 = QtWidgets.QLabel(MainWindow)
    #     self.label_3.setGeometry(QtCore.QRect(20, 180, 179, 221))
    #     self.label_3.setText("")
    #     self.label_3.setObjectName("label_face")
    #
    #     self.retranslateUi(MainWindow)
    #
    #     MainWindow.setCentralWidget(self.centralWidget)
    #     QtCore.QMetaObject.connectSlotsByName(MainWindow)
    #
    #     self.btn_input.clicked.connect(self.openfile)
    #     self.btn_output.clicked.connect(self.do_recognition)
    #
    # def retranslateUi(self, MainWindow):
    #     _translate = QtCore.QCoreApplication.translate
    #     MainWindow.setWindowTitle(_translate("MainWindow", "身份证照片矫正"))
    #     MainWindow.setWindowIcon(QIcon("../images/身份证带圈.png"))
    #     self.btn_input.setText(_translate("MainWindow", "选择图片"))
    #     self.btn_output.setText(_translate("MainWindow", "图像校正"))
    #     self.label.setText(_translate("MainWindow", "图片路径："))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())
