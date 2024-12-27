import torch
import torch.nn as nn
import torch.optim as optim
from ..models.cnn import EMNISTCNN
from ..utils.preprocess import train_loader, test_loader

def train_model(model, optimizer, criterion, device, epochs=5):
    print('Training emnist cnn ...') 
    model.train
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backwards()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')
        
def test_model(model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print(f'Test Accuracy: {100 * correct /  total:.2f}%')
    
if __name__ == '__main__':
    model = EMNISTCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model(model, optimizer, criterion, device, epoch=5)
    test_model(model, device)
    save_path = '../trained_models/emnist_cnn.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')