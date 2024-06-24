# model.py
import torch.nn as nn
import torchvision.models as models

class ViolenceClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ViolenceClassifier, self).__init__()
        # 使用 weights 参数加载预训练权重
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
