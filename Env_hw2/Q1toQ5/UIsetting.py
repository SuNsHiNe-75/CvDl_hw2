# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QPoint
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 880)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # for Q5
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(490, 450, 601, 401))
        self.groupBox_7.setObjectName("groupBox_7")
        self.pushButton_16 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_16.setGeometry(QtCore.QRect(20, 30, 200, 40))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_17 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_17.setGeometry(QtCore.QRect(20, 100, 200, 40))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_18.setGeometry(QtCore.QRect(20, 170, 200, 40))
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_19 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_19.setGeometry(QtCore.QRect(20, 240, 200, 40))
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_20 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_20.setGeometry(QtCore.QRect(20, 310, 200, 40))
        self.pushButton_20.setObjectName("pushButton_20")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_7)
        self.graphicsView.setGeometry(QtCore.QRect(260, 40, 301, 281))
        self.graphicsView.setObjectName("graphicsView")
        self.label_3 = QtWidgets.QLabel(self.groupBox_7)
        self.label_3.setGeometry(QtCore.QRect(260, 40, 301, 281))
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.groupBox_7)
        self.label.setGeometry(QtCore.QRect(280, 340, 91, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_7)
        self.label_2.setGeometry(QtCore.QRect(380, 340, 121, 21))
        self.label_2.setObjectName("label_2")


        # for Q4
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(490, 20, 601, 401))
        self.groupBox_6.setObjectName("groupBox_6")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_12.setGeometry(QtCore.QRect(20, 50, 221, 40))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_13.setGeometry(QtCore.QRect(20, 130, 221, 40))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_14.setGeometry(QtCore.QRect(20, 210, 221, 40))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_15.setGeometry(QtCore.QRect(20, 290, 221, 40))
        self.pushButton_15.setObjectName("pushButton_15")
        self.textArea = QtWidgets.QTextBrowser(self.groupBox_6)
        self.textArea.setGeometry(QtCore.QRect(20, 350, 221, 35))
        self.textArea.setObjectName("textArea")

        #繪圖部分  創建儲存按鈕
        self.paintWidget = HandwritingBoard()
        self.layoutForMNIST = QHBoxLayout()
        self.layoutForMNIST.addStretch()
        self.layoutForMNIST.addWidget(self.paintWidget)
        self.pushButton_14.clicked.connect(self.onPredictButtonClicked)
        self.pushButton_15.clicked.connect(self.paintWidget.Reset)       
        # self.pushButtonShowInference1.clicked.connect(self.paintWidget.save)      
        
        # 創建 MNIST GroupBox 並將水平布局應用到其中
        self.MNIST = QtWidgets.QGroupBox(self.groupBox_6)
        self.MNIST.setGeometry(QtCore.QRect(265, 70, 311, 271))
        self.MNIST.setLayout(self.layoutForMNIST)


        # for Q3
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(220, 390, 240, 111))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_11.setGeometry(QtCore.QRect(20, 40, 200, 40))
        self.pushButton_11.setObjectName("pushButton_11")

        # for Q2
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(220, 170, 240, 181))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_9.setGeometry(QtCore.QRect(20, 40, 200, 40))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_10.setGeometry(QtCore.QRect(20, 110, 200, 40))
        self.pushButton_10.setObjectName("pushButton_10")

        # for Q1
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(219, 20, 241, 111))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 40, 200, 40))
        self.pushButton_4.setObjectName("pushButton_4")

        # for Load Image & Video
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 171, 480))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 90, 131, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 220, 131, 40))
        self.pushButton_2.setObjectName("pushButton_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        # for Q5
        self.groupBox_7.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.pushButton_16.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_17.setText(_translate("MainWindow", "5.1 Show Images"))
        self.pushButton_18.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.pushButton_19.setText(_translate("MainWindow", "5.3 Show Comparison"))
        self.pushButton_20.setText(_translate("MainWindow", "5.4 Inference"))
        self.label.setText(_translate("MainWindow", "Predicted ="))

        # for Q4
        self.groupBox_6.setTitle(_translate("MainWindow", "4. MNIST Classifier using VGG19"))
        self.pushButton_12.setText(_translate("MainWindow", "4.1 Show Model Structure"))
        self.pushButton_13.setText(_translate("MainWindow", "4.2 Show Accuracy and Loss"))
        self.pushButton_14.setText(_translate("MainWindow", "4.3 Predict"))
        self.pushButton_15.setText(_translate("MainWindow", "4.4 Reset"))

        # for Q3
        self.groupBox_5.setTitle(_translate("MainWindow", "3. PCA"))
        self.pushButton_11.setText(_translate("MainWindow", "3. Dimension Reduction"))

        # for Q2
        self.groupBox_4.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.pushButton_9.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.pushButton_10.setText(_translate("MainWindow", "2.2 Video Tracking"))

        # for Q1
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.pushButton_4.setText(_translate("MainWindow", "1. Background Subtraction"))

        # for load Image
        self.pushButton.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Video"))

    def showPredictionResult(self):
        if self.paintWidget.predicted_class is not None:
            print(self.paintWidget.predicted_class)
            self.textArea.setText(f'The predicted digit is: {self.paintWidget.predicted_class}')
        else:
            self.textArea.clear()

    def onPredictButtonClicked(self):
        # 先執行 SaveAndPredict
        self.paintWidget.SaveAndPredict()
        # 再執行 showPredictionResult
        self.showPredictionResult()

#繪圖部分
class HandwritingBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.predicted_class = None

    def initUI(self):
        self.setFixedSize(350, 350)
        self.setStyleSheet("background-color: black;")
        self.image = QPixmap(350, 350)
        self.image.fill(Qt.black)
        self.lastPoint = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(QColor("white"), 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def SaveAndPredict(self):
        # 保存寫字板的內容為圖片
        self.image.save("./handwrite.png")
        imgPath = "./handwrite.png"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        VGG19_BN_model = torch.load("./VGG19_BN_model.pth", map_location=torch.device('cpu'))
        VGG19_BN_model = VGG19_BN_model.to(device)

        # 將模型切換為評估模式
        VGG19_BN_model.eval()

        # 載入並預處理單張圖片
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 需要根據模型的輸入尺寸進行調整
            transforms.Grayscale(num_output_channels=1),  # 將圖像轉換為單通道
            transforms.ToTensor(),
        ]   )

        img = Image.open(imgPath)
        input_tensor = transform(img).unsqueeze(0)

        # 如果你的模型在 GPU 上，將輸入數據移動到 GPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        VGG19_BN_model = VGG19_BN_model.to(device)

        # 進行預測
        with torch.no_grad():
            output = VGG19_BN_model(input_tensor)

        # 處理預測結果
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        self.predicted_class = torch.argmax(probabilities).item()

        print(f"The predicted digit is: {self.predicted_class}")
        print(f"Class probabilities: {probabilities}")

        # 顯示機率分佈的直方圖
        plt.figure()
        plt.bar(range(10), probabilities.tolist(), tick_label=range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.show()


    
    def Reset(self):
        self.image.fill(Qt.black)
        self.update()