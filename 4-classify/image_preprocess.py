#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

# 定义图像转换过程，包括数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并调整到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转10度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 图像预处理函数
def preprocess_images(image_folder):
    images = []
    labels = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
            image = transform(image)
            label = 0 if image_name.startswith('0') else 1
            images.append(image)
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)

# 预处理train和val文件夹中的图像
train_folder = 'train'  # 替换为你的train文件夹路径
val_folder = 'val'  # 替换为你的val文件夹路径

train_images, train_labels = preprocess_images(train_folder)
val_images, val_labels = preprocess_images(val_folder)

# 保存预处理后的Tensor
torch.save((train_images, train_labels), 'preprocessed_train.pt')
torch.save((val_images, val_labels), 'preprocessed_val.pt')


# In[8]:


import torch
from torch.utils.data import Dataset, DataLoader

class ViolenceDataset(Dataset):
    def __init__(self, data_path):
        self.images, self.labels = torch.load(data_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 加载预处理后的数据
train_dataset = ViolenceDataset('preprocessed_train.pt')
val_dataset = ViolenceDataset('preprocessed_val.pt')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# In[11]:


import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 224/2/2 = 56
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[12]:


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        print(labels)


# In[ ]:




