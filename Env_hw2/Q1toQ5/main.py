from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import os
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from UIsetting import Ui_MainWindow
from datetime import datetime
from sklearn.decomposition import PCA
from Q5 import Question5

def concath(list_array):
    return cv2.hconcat(list_array)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
		# in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        #TODO
        self.ui.pushButton_2.clicked.connect(self.load_video)
        self.ui.pushButton_4.clicked.connect(self.backgroundSubtraction)
        self.ui.pushButton_9.clicked.connect(self.preprocessing)
        self.ui.pushButton_10.clicked.connect(self.vedioTracking)

        self.ui.pushButton.clicked.connect(self.load_Image_PCA)
        self.ui.pushButton_11.clicked.connect(self.dimensionReduction)

        self.ui.pushButton_12.clicked.connect(self.showModelStructure)
        self.ui.pushButton_13.clicked.connect(self.showAccuracyAndLoss)
        # pushButton_14/15 are in UIsetting.
        self.ui.pushButton_16.clicked.connect(self.load_Image)
        self.ui.pushButton_17.clicked.connect(Q5Object.showImages)
        self.ui.pushButton_18.clicked.connect(Q5Object.showModelStructure)
        self.ui.pushButton_19.clicked.connect(Q5Object.showComparison)
        self.ui.pushButton_20.clicked.connect(self.showInference)
    
    #1
    def load_video(self):
        self.video = QFileDialog.getOpenFileName(self)

    def backgroundSubtraction(self):

        now = datetime.now()
        # convert to string
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        """
        Inital video
        """
        # cap = cv2.VideoCapture("Dataset_CvDl_Hw2/Q1/traffic.mp4")
        cap = cv2.VideoCapture(self.video[0])
        frames = []
        build_model = False
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print((w, h))
        # video recorder
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv.CAP_PROP_FOURCC(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
        video_writer = cv2.VideoWriter("output.avi",fourcc, fps, (w*3, h))
        while cap.isOpened():
            "Read video frame"
            ret, frame = cap.read()
            if ret:
                "Convert BGR frame to GRAY frame"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(gray)
                "Get first 25 frames for building Gaussian model"
                if len(frames) < 25:
                    frames.append(gray)
                else:
                    if not build_model:
                        frames = np.array(frames)
                        """For every pixels in video from 0~25 frames, 
                        build a gaussian model with mean and standard deviation 
                        (if standard deviation is less then 5, set to 5)
                        """
                        mean = np.mean(frames, axis= 0)
                        standard = np.std(frames, axis=0)
                        standard[standard < 5] = 5
                        build_model = True
                    else:
                        """
                        For frame > 50, test every frame pixels with respective gaussian model. 
                        If gray value difference between testing pixel and gaussian mean 
                        is larger than 5 times standard deviation, 
                        set testing pixel to 255 (foreground, white), 0 (background, black) otherwise.
                        """
                        mask[np.abs(gray - mean) > standard*5] = 255
                """
                Show the result
                """
                foreground = cv2.bitwise_and(frame, frame, mask= mask)
                mask_out = np.zeros_like(frame)
                mask_out[:,:,0] = mask
                mask_out[:,:,1] = mask
                mask_out[:,:,2] = mask

                out = concath([frame, mask_out, foreground])
                video_writer.write(out)
                cv2.imshow("Video", out)
                key = cv2.waitKey(10)
                if key == 27:
                    break
                elif key == ord("c"):
                    path = os.path.join(os.getcwd(), "Screenshot_{}.png".format(date_time_str))
                    cv2.imwrite(path, frame)
            else:
                break
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


    #2
    def preprocessing(self):
        # 讀取影片
        cap = cv2.VideoCapture('./Dataset_CvDl_Hw2/Q2/optical_flow.mp4')

        # 創建Shi-Tomasi角點檢測的參數
        feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        # 創建紅色顏色
        color = (0, 0, 255)

        # 讀取第一幀
        ret, first_frame = cap.read()

        # 轉換成灰度圖
        gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # 使用goodFeaturesToTrack檢測角點
        p0 = cv2.goodFeaturesToTrack(gray_first_frame, mask=None, **feature_params)

        # 取第一個角點座標
        x, y = map(int, p0[0].ravel())

        # 在第一幀上畫出紅色十字標記
        cv2.line(first_frame, (x - 20, y), (x + 20, y), color, 4)
        cv2.line(first_frame, (x, y - 20), (x, y + 20), color, 4)

        # 等比例縮小 0.5 倍
        # scaled_frame = cv2.resize(first_frame, (0, 0), fx=0.5, fy=0.5)

        # 使用 matplotlib 顯示圖片
        plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 不顯示座標軸
        plt.show()

        cap.release()

        self.detected_point = x, y

    def vedioTracking(self):
        cap = cv2.VideoCapture('./Dataset_CvDl_Hw2/Q2/optical_flow.mp4')

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 1,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        color_yellow = (0, 255, 255)  # 這是黃色的 RGB 值
        color_red = (0, 0, 255)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            # 取第一個角點座標
            x, y = map(int, p0[0].ravel())

            # 在第一幀上畫出紅色十字標記
            cv2.line(frame, (x - 20, y), (x + 20, y), color_red, 4)
            cv2.line(frame, (x, y - 20), (x, y + 20), color_red, 4)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color_yellow, 4)
            
            img = cv2.add(frame, mask)
            img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('2-2 Optimal Flow', img_resized)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()   


    #3
    def load_Image_PCA(self):
        self.imgPCA = QFileDialog.getOpenFileName(self)

    def dimensionReduction(self):
        if self.imgPCA[0] == None: 
            print('Please load the image.')

        else:
            image = cv2.imread(self.imgPCA[0]) 

            # Convert RGB image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize grayscale image
            normalized_image = gray_image / 255.0

            # Perform PCA
            height, width = normalized_image.shape
            min_dimension = min(height, width)

            # Initialize variables
            n = 1
            reconstruction_error = float('inf')

            # Loop to find minimum components with reconstruction error less than or equal to 3.0
            while reconstruction_error > 3.0 and n <= min_dimension:
                pca = PCA(n_components=n)
                pca.fit(normalized_image)
                components = pca.transform(normalized_image)
                reconstructed_image = pca.inverse_transform(components)
                reconstruction_error = np.mean((normalized_image - reconstructed_image) ** 2)
                n += 50

            # Print the minimum n value
            print("Minimum n value with reconstruction error <= 3.0:", n - 1)

            # Perform PCA again for the specified n value
            pca = PCA(n_components=n - 1)
            pca.fit(normalized_image)
            components = pca.transform(normalized_image)
            reconstructed_image = pca.inverse_transform(components)

            # Plot the grayscale image and the reconstruction image
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.title('Gray scale Image')
            plt.imshow(normalized_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f'Reconstructed Image (n={n-1})')
            plt.imshow(reconstructed_image, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


    #4
    def showModelStructure(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        VGG19_BN_model = torch.load("./VGG19_BN_model.pth", map_location=torch.device('cpu'))
        VGG19_BN_model = VGG19_BN_model.to(device)
        summary(VGG19_BN_model, input_size=(1, 224, 224), device = 'cpu')
    
    def showAccuracyAndLoss(self):
        image = plt.imread("./combined_metrics_plot.png")

        # Display the image
        plt.figure(num="Accuracy and Loss")
        plt.axis("off")
        plt.imshow(image)
        plt.show()


    #5
    def load_Image(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='D:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get tuple[0] which is file name
        self.imagePath = fileName
        myPixmap = QtGui.QPixmap(self.imagePath)
        myScaledPixmap = myPixmap.scaled(self.ui.label_3.size(), Qt.KeepAspectRatio)
        self.ui.label_3.setPixmap(myScaledPixmap)

    def showInference(self):
        label = Q5Object.showInference(self.imagePath)

        if label:
            self.ui.label_2.setText(label)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Q5Object = Question5()

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())