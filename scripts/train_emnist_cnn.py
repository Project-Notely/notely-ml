import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import EMNISTCNN
from utils.preprocess import EMNISTPreprocessor


def train_model(model, train_loader, optimizer, criterion, device, epochs=5):
    print('Training EMNIST CNN ...')
    start_time = time.time()
    model.train
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )
                
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(
            f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, '
            f'Time: {epoch_time:.2f}s'
        )
        
        # save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = 'trained_models/emnist_cnn_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved with loss: {epoch_loss:.4f}')
        
    total_time = time.time() - start_time
    print(f'\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')


def test_model(model, test_loader, device):
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
    preprocessor = EMNISTPreprocessor('./data', 64)
    train_loader = preprocessor.train_loader
    test_loader = preprocessor.test_loader

    train_model(model, train_loader, optimizer, criterion, device, epochs=10)
    test_model(model, test_loader, device)