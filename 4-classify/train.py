import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from data_loader import get_loaders
from model import SimpleCNN

def validate(model, val_loader, criterion, device):
    model.eval()  # Sets the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Deactivates autograd
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%')

    print("Training complete.")

if __name__ == '__main__':
    train_loader, val_loader = get_loaders('/home/kali/Desktop/work/violence_224/train', '/home/kali/Desktop/work/violence_224/val', batch_size=32)
    model = SimpleCNN()
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

