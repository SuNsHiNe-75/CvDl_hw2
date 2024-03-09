import matplotlib.pyplot as plt
import glob
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchsummary import summary

class Question5:
    def __init__(self) -> None:
        pass

    def showImages(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), num="Resized images in inference dataset")

        catInferenceDataset = glob.glob(".\\dataset\\inference_dataset\\Cat\\*.jpg")
        dogInferenceDataset = glob.glob(".\\dataset\\inference_dataset\\Dog\\*.jpg")

        imgCat = Image.open(random.choice(catInferenceDataset))
        imgCat = imgCat.resize((224, 224))

        imgDog = Image.open(random.choice(dogInferenceDataset))
        imgDog = imgDog.resize((224, 224))

        axes[0].imshow(imgCat)
        axes[0].set_title("Cat")
        axes[0].axis("off")

        axes[1].imshow(imgDog)
        axes[1].set_title("Dog")
        axes[1].axis("off")

        plt.show()

    def showModelStructure(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet50_model = torch.load("./model_40.pt", map_location=torch.device('cpu'))
        resnet50_model = resnet50_model.to(device)
        summary(resnet50_model, input_size=(3, 224, 224), device = 'cpu')
    
    def showComparison(self):
        # make a comparison picture
        # self.compareModels()

        image = plt.imread("model_comparison.png")

        # Display the image
        plt.figure(num="Model Comparison")
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def showInference(self, imgPath):

        if imgPath == None:
            print("please load the image which want to predict")
            return

        print("Predicting...")

        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        model = torch.load("./model_40.pt", map_location=torch.device('cpu'))

        # 將模型切換為評估模式
        model.eval()

        # 載入並預處理單張圖片
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 需要根據模型的輸入尺寸進行調整
            transforms.ToTensor(),
        ]   )

        img = Image.open(imgPath)
        input_tensor = transform(img).unsqueeze(0)

        # 如果你的模型在 GPU 上，將輸入數據移動到 GPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        model = model.to(device)

        # 進行預測
        with torch.no_grad():
            output = model(input_tensor)

        # print(output)
        threshold = 0.5
        predictions = (output > threshold).float()
        # print(predictions)
        result = predictions.item()
        # print(result)

        if result < 0.5:
            predOutcome = 'Cat'
        else :
            predOutcome = 'Dog'

        print(predOutcome)

        return predOutcome
        # return predOutcome
