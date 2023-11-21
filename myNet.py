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


# class 1=>584, class 0=>32542
class MyNewDataset(Dataset):
    def __init__(self,label_csv,root_dir,transforms=None):
        self.img_labels=pd.read_csv(label_csv)
        self.root_dir=root_dir
        self.transforms=transforms
        self.target=torch.tensor(self.img_labels.iloc[:, 1])

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

# Define data augmentation transform
augmentation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    
    ##
    ##
    # Add other data augmentation operations
    transforms.ToTensor(),
])

# Define my dataset

label_csv = 'modified1.csv'
trainFolder='real-resized'
#train_data_origin = MyDataset(filepath=train_file, transforms=augmentation_transform)
myData = MyNewDataset(label_csv=label_csv,root_dir=trainFolder,transforms=augmentation_transform)
#train_data = oversample_data(train_data_origin)
generator1 = torch.Generator().manual_seed(42)
train_data,val_data=torch.utils.data.random_split(myData,[0.9,0.1],generator1)
train_loader = DataLoader(dataset=train_data,
                          batch_size=25,
                          shuffle=True,
                          drop_last=True)


test_lst=[0,1]
val_loader=DataLoader(dataset=val_data,
                        batch_size=25,
                        shuffle=True,
                        drop_last=True)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Use_gpu = torch.cuda.is_available()
model = models.resnet18(pretrained=True)
print(f"Origine Model= {model}")

for param in model.parameters():
    param.requires_grad = True 

model.fc=torch.nn.Linear(512,2)
print(f"Modified Model= {model}")

if Use_gpu:
    model = model.to(device)


#Loss Fonction & Modulation of weight
weights = [0.5, 0.5]
class_weights = torch.FloatTensor(weights).to(device)
loss_func = nn.CrossEntropyLoss(weight=class_weights)
#loss_func = nn.CrossEntropyLoss()

#Learning rate=0.01
learning_rate=0.001
#Optimizer
optimizer_train=torch.optim.Adam(model.parameters(),lr=learning_rate)
# optimizer_train=torch.optim.Adam(cifar_train.parameters(),lr=learning_rate)
#Setting Steps
total_train_step=0
total_test_step=0
#testDataset
testFolder='test-resized'
testLabel='output.csv'
finalResult=[]
myTestData = MyNewDataset(label_csv=testLabel,root_dir=testFolder)
testDataloader=DataLoader(dataset=myTestData,
                        batch_size=64,
                        shuffle=False,
                        drop_last=False)

#Rounds for training
epoch=40

#data visualize: TensorBoard
writer=SummaryWriter("./logs_myDataset")

for i in range(epoch):
    print("-----------------The {} round start-------------------".format(i+1))

    #test result total of every epoch
    final_predict=[]
    final_true=[]
    #start training
    for j,data in enumerate(train_loader):
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        # imgs=imgs.reshape(20,3,74,74)  #batch,channel,img.height, img.weight
        outputs=model(imgs)

        #Add class weight to loss function
        loss=loss_func(outputs,targets.long())
        # print(f"loss={loss}")

        #optimizer
        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("Train step {}, Loss: {}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss,total_train_step)

    #start validating
    # Create model
    target_name=[str(i) for i in test_lst]
    # print(f"target_name= {target_name}")
    with torch.no_grad():    #ensure this data would not be optimized
        for l,data in enumerate(val_loader):
            imgs,targets=data
            imgs,targets=imgs.to(device),targets.to(device)
            outputs=model(imgs)
            # outputs = cifar_train(imgs)

            #loss=loss_func(outputs)
            outputs_np=outputs.cpu().numpy()
            #Decide the prediction of outputs --[predict probability of each class]
            x_predict=np.argmax(outputs_np,axis=1)
            x_predict=x_predict.tolist()
            #Get the true labels
            targets_np=targets.cpu().numpy()
            targets_np=targets_np
            targets_ls=targets_np.tolist()
            targets_ls=list(map(int,targets_ls[:]))
            y_true = list()
            for q in targets_ls:
                t=test_lst.index(q)
                y_true.append(t)

            #Sum up all batches of label lists
            final_true.extend(y_true)
            final_predict.extend(x_predict)

    #print(f'final predict label list= {final_predict}')
    #print(f'final true label list= {final_true}')
    #report table fonction
    #report table fonction
    total_test_step = total_test_step + 1
    report= classification_report(final_true, final_predict)
    report_dict = classification_report(final_true, final_predict,output_dict=True)
    print(report)
    #print(report_dict)
    #record result in tensorboard
    writer.add_scalar("0_precision", report_dict['0']['precision'] , total_test_step)
    writer.add_scalar("1_precision", report_dict['1']['precision'], total_test_step)
    writer.add_scalar("0_recall", report_dict['0']['recall'], total_test_step)
    writer.add_scalar("1_recall", report_dict['1']['recall'], total_test_step)
    writer.add_scalar("0_f1-score", report_dict['0']['f1-score'], total_test_step)
    writer.add_scalar("1_f1-score", report_dict['1']['f1-score'], total_test_step)
    writer.add_scalar("total-accuracy", report_dict['accuracy'], total_test_step)

    # save trained model
    torch.save(model,"model_{}.pth".format(i))
    print("Model saved...")


'''
# start testing

print("---------------start testing-----------------")
with torch.no_grad():    #ensure this data would not be optimized
    for _,data in enumerate(testDataloader):
        imgs,_=data
        imgs=imgs.to(device)
        #print(testDataloader)
        outputs=model(imgs.float())
        outputs_np=outputs.cpu().numpy()
        #Decide the prediction of outputs --[predict probability of each class]
        x_predict=np.argmax(outputs_np,axis=1)
        x_predict=x_predict.tolist()
        finalResult.extend(x_predict)
resultCSV='result.csv'

outputCSV(testFolder,resultCSV,finalResult)
'''

writer.close()
