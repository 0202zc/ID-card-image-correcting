# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'recog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import sys

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(824, 493)
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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "身份证照片矫正"))
        self.label.setText(_translate("MainWindow", "图像路径："))
        self.btn_input.setText(_translate("MainWindow", "选择图片"))
        self.btn_output.setText(_translate("MainWindow", "图像矫正"))
        self.btn_pre.setText(_translate("MainWindow", "图像预处理"))
        self.btn_edge.setText(_translate("MainWindow", "边缘提取"))
        self.btn_trans.setText(_translate("MainWindow", "透视变换"))
        self.btn_fill.setText(_translate("MainWindow", "边缘填充"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())
