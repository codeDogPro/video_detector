# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.horizontalLayout_5.addWidget(self.title)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self._2 = QtWidgets.QVBoxLayout()
        self._2.setObjectName("_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.import_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.import_btn.setFont(font)
        self.import_btn.setObjectName("import_btn")
        self.horizontalLayout.addWidget(self.import_btn)
        self.file_path = QtWidgets.QLineEdit(self.centralwidget)
        self.file_path.setObjectName("file_path")
        self.horizontalLayout.addWidget(self.file_path)
        self._2.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.open_camera_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.open_camera_btn.setFont(font)
        self.open_camera_btn.setObjectName("open_camera_btn")
        self.horizontalLayout_7.addWidget(self.open_camera_btn)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self._2.addLayout(self.horizontalLayout_7)
        self.config_vl = QtWidgets.QVBoxLayout()
        self.config_vl.setContentsMargins(-1, -1, -1, 20)
        self.config_vl.setObjectName("config_vl")
        self.config_lb = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.config_lb.setFont(font)
        self.config_lb.setObjectName("config_lb")
        self.config_vl.addWidget(self.config_lb)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.iou_lb = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.iou_lb.setFont(font)
        self.iou_lb.setObjectName("iou_lb")
        self.horizontalLayout_2.addWidget(self.iou_lb)
        self.score_value = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.score_value.setFont(font)
        self.score_value.setMaximum(1.0)
        self.score_value.setSingleStep(0.01)
        self.score_value.setProperty("value", 0.6)
        self.score_value.setObjectName("score_value")
        self.horizontalLayout_2.addWidget(self.score_value)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.config_vl.addLayout(self.horizontalLayout_2)
        self._2.addLayout(self.config_vl)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.start_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.start_btn.setFont(font)
        self.start_btn.setObjectName("start_btn")
        self.horizontalLayout_4.addWidget(self.start_btn)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.stop_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stop_btn.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.stop_btn.setFont(font)
        self.stop_btn.setObjectName("stop_btn")
        self.horizontalLayout_4.addWidget(self.stop_btn)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self._2.addLayout(self.horizontalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, 20, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.result_lb = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.result_lb.setFont(font)
        self.result_lb.setObjectName("result_lb")
        self.verticalLayout.addWidget(self.result_lb)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.object_num_lb = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.object_num_lb.setFont(font)
        self.object_num_lb.setObjectName("object_num_lb")
        self.horizontalLayout_3.addWidget(self.object_num_lb)
        self.num_lb = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.num_lb.setFont(font)
        self.num_lb.setObjectName("num_lb")
        self.horizontalLayout_3.addWidget(self.num_lb)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.result_show = QtWidgets.QListView(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("思源黑体 CN")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.result_show.setFont(font)
        self.result_show.setObjectName("result_show")
        self.verticalLayout.addWidget(self.result_show)
        self._2.addLayout(self.verticalLayout)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.LabelRole, self._2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setText("")
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label)
        self.verticalLayout_2.addLayout(self.formLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "视频目标检测系统"))
        self.import_btn.setText(_translate("MainWindow", "导入视频"))
        self.open_camera_btn.setText(_translate("MainWindow", "打开摄像头"))
        self.config_lb.setText(_translate("MainWindow", "配置："))
        self.iou_lb.setText(_translate("MainWindow", "Score 阈值："))
        self.start_btn.setText(_translate("MainWindow", "开始"))
        self.stop_btn.setText(_translate("MainWindow", "暂停"))
        self.result_lb.setText(_translate("MainWindow", "检测结果："))
        self.object_num_lb.setText(_translate("MainWindow", "目标数目："))
        self.num_lb.setText(_translate("MainWindow", "0/0"))
        self.label_3.setText(_translate("MainWindow", "类别：Score"))
