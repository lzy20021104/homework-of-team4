# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data, self.labels = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

# 使用DataLoader加载数据
def get_data_loader(data_path, batch_size=32, shuffle=True, num_workers=4):
    dataset = CustomDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
