#!/usr/bin/env python
# coding: utf-8

import os
from PIL import Image
import torch
from torchvision import transforms

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
            label = 0 if image_name.startswith('0') else 1  # 根据文件名生成标签
            images.append(image)
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)

# 预处理图像并保存到.pt文件
def preprocess_and_save(image_folder, save_path):
    print(f"Preprocessing images in {image_folder}...")
    train_images, train_labels = preprocess_images(image_folder)
    print(f"Saving preprocessed data to {save_path}...")
    torch.save((train_images, train_labels), save_path)
    print("Done.")

# 指定训练集和验证集的文件夹路径
train_folder = '/home/pjf/Desktop/try/train1'  # 替换为训练集文件夹的路径
val_folder = '/home/pjf/Desktop/try/val1'      # 替换为验证集文件夹的路径

# 指定保存预处理后数据的文件路径
train_save_path = 'preprocessed_train.pt'
val_save_path = 'preprocessed_val.pt'

# 执行预处理并保存数据
preprocess_and_save(train_folder, train_save_path)
preprocess_and_save(val_folder, val_save_path)
