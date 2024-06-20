# data_loader.py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ViolenceDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.jpg')]
        self.labels = [0 if '0_' in fname else 1 for fname in os.listdir(directory) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_loaders(train_dir, val_dir, batch_size):
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ViolenceDataset(train_dir, transform=transformations)
    val_dataset = ViolenceDataset(val_dir, transform=transformations)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 测试数据加载器
if __name__ == "__main__":
    train_loader, val_loader = get_loaders('/home/kali/Desktop/work/violence_224/train', '/home/kali/Desktop/work/violence_224/val', 32)

    for images, labels in train_loader:
        print(f"Loaded {len(images)} images")
        print(f"Sample labels: {labels[:5]}")
        break
