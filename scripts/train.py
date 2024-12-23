import time
from datetime import datetime, timedelta
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.recognition import HandwrittingCNN
from src.utils.preprocess import EMNISTPreprocessor

def format_time(seconds):
    """convert seconds to human readable string"""
    return str(timedelta(seconds=int(seconds)))

# record start time
start_time = time.time()

# init preprocessor and get data loaders
preprocessor = EMNISTPreprocessor()
train_loader, test_loader = preprocessor.load_datasets()

# init the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwrittingCNN(num_classes=62).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# training loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


# testing loop
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# saving model
def save_checkpoint(model, optimizer, epoch, accuracy, path='checkpoint'):
    import os
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, f'{path}/model_epoch_{epoch}.pth')
    
    
# training loop with progress tracking
print(f'Using device: {device}')
print('Starting training...')
    
epochs = 10
best_accuracy = 0.0
epoch_times = []

for epoch in range(epochs):
    epoch_start = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_acc = test(model, test_loader, device)
    
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Time taken: {format_time(epoch_time)}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        save_checkpoint(model, optimizer, epoch+1, best_accuracy)
        print(f'Saved new best model with accuracy: {best_accuracy:.4f}')
        
    print('-'*50)

# calculate final timing statistics
total_time = time.time() - start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)


print("Training complete!")
print(f"Best accuracy achieved: {best_accuracy:.4f}")
