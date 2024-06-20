如何使用`classify.py`文件中的接口类来进行不良内容图像的检测：

前提条件：

已正确安装了PyTorch和torchvision库，并且准备好预训练的模型文件。

简单讲解：

data_loader创建一个自定义的数据集类 ViolenceDataset 和一个函数 get_loaders，用于加载和预处理图像数据，并生成训练集和验证集的数据加载器。
代码还包含了一个简单的测试部分，用于验证数据加载器的功能。
调用时首先导入get_loaders函数，定义训练和验证数据集的目录路径以及批次大小。
接下来，调用get_loaders函数并传入这些参数，得到训练数据加载器train_loader和验证数据加载器val_loader。

model定义了一个简单的卷积神经网络模型，并在文件的末尾进行了测试。
创建一个SimpleCNN的实例，并打印出模型的结构，以查看模型的详细信息。

train从数据加载到模型定义，再到训练和验证，都通过函数进行模块化处理。
同时打印训练和验证过程中的损失和准确率。
train代码中/home/kali/Desktop/work/violence_224/train处地址可视情况进行修改

