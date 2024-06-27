from torchvision import models  
import torchvision.models as models  
import torch.nn as nn

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from model import ViolenceClassifier
from data_loader import CustomDataset

class ViolenceClassifier(nn.Module):  
    def __init__(self, num_classes):  # 添加 num_classes 参数来指定输出层的大小  
        super(ViolenceClassifier, self).__init__()  
        # 使用 pretrained=True 来加载带有预训练权重的 ResNet18 模型  
        self.resnet = models.resnet18(pretrained=True)  
          
        # 假设你想要修改 ResNet18 的最后一层以适应你的分类任务  
        # 你可以通过修改全连接层来改变输出类别的数量  
        num_ftrs = self.resnet.fc.in_features  # 获取原全连接层的输入特征数  
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # 替换全连接层  
  
    def forward(self, x):  
        # 直接调用 ResNet18 的前向传播  
        return self.resnet(x)  
  
# 实例化模型时，指定分类任务的类别数量  
num_classes = 10  # 假设你有10个类别  
model = ViolenceClassifier(num_classes)

# 准确率计算函数
def calculate_accuracy(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 模型训练函数
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # 验证集评估
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, "
              f"Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 检查是否达到最佳验证准确率
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best accuracy achieved: {best_accuracy:.4f} at epoch {epoch+1}")

if __name__ == "__main__":
    # 数据加载器初始化
    train_dataset = CustomDataset("preprocessed_train.pt")
    val_dataset = CustomDataset("preprocessed_val.pt")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
import torchvision.models as models
# 假设 train_model 函数和 train_loader, val_loader 已经在其他地方定义好了

    
# 训练模型
train_model(model, train_loader, val_loader, epochs=10, lr=0.001)
