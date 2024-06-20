# model.py
import torch  # 添加这行来导入torch模块
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 在 model.py 中测试模型
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)  # 打印模型结构
    # 测试一个随机数据批次
    inputs = torch.rand(5, 3, 224, 224)  # 假设有5个样本，每个样本是224x224的3通道图像
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")
