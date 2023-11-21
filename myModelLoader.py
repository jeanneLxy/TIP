import csv
import os
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.io import read_image
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# function to output csv file
def outputCSV(folder_path, output_csv, data_list):
    files = os.listdir(folder_path)
    with open(output_csv, 'w', newline='') as csvfile:
        # 创建 CSV writer，指定分隔符为逗号
        csv_writer = csv.writer(csvfile, delimiter=',')
        # 写入 CSV 文件的标题（假设为 "image_path" 和 "label"）
        csv_writer.writerow(['image_name', 'target'])
        # 遍历文件夹中的文件并将文件名（去掉后缀）和列表元素写入 CSV 文件
        for file in files:
            # 使用 os.path.splitext 获取文件名和扩展名
            file_name, file_extension = os.path.splitext(file)
            # 检查文件扩展名是否为 '.jpg'
            if file_extension.lower() == '.jpg':
                # 获取列表元素，如果列表为空或长度不够，则使用默认值 0
                label = data_list.pop(0) if data_list else 0
                # 将文件名（去掉后缀）和列表元素写入 CSV 文件的一行
                csv_writer.writerow([file_name, label])

class MyNewDataset(Dataset):
    def __init__(self,label_csv,root_dir,transforms=None):
        self.img_labels=pd.read_csv(label_csv)
        self.root_dir=root_dir
        self.transforms=transforms

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, item):
        img_path=os.path.join(self.root_dir,self.img_labels.iloc[item,0])
        img_path+='.jpg'
        imgs=read_image(img_path)
        label=self.img_labels.iloc[item,1]
        if self.transforms:
            imgs=self.transforms(imgs)
        return imgs,label

#load model
model = torch.load('model_14.pth')
print(model)
# Put the model in evaluation mode
#model.eval()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.to(device)

#testDataset
testFolder='test-resized'
testLabel='output.csv'
finalResult=[]
myTestData = MyNewDataset(label_csv=testLabel,root_dir=testFolder)
testDataloader=DataLoader(dataset=myTestData,
                        batch_size=25,
                        shuffle=False,
                        drop_last=False)

with torch.no_grad():    #ensure this data would not be optimized
    for _,data in enumerate(testDataloader):
        imgs,_=data
        imgs=imgs.to(device)
        #print(testDataloader)
        outputs=model(imgs.float())
        outputs_np=outputs.cpu().numpy()
        #print(outputs)
        #Decide the prediction of outputs --[predict probability of each class]
        x_predict=np.argmax(outputs_np,axis=1)
        
        x_predict=x_predict.tolist()
        finalResult.extend(x_predict)
        #print(finalResult)
resultCSV='result.csv'

outputCSV(testFolder,resultCSV,finalResult)
