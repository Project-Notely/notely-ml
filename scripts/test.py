import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool2(self.relu2(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.relu3(self.fc1(x))  # Fully connected layer -> ReLU
        x = self.fc2(x)  # Final layer
        return x

# Prepare the dataset and data loaders
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

# Train the model
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')

# Test the model
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

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Load the model
def load_model(path, device):
    model = CNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f'Model loaded from {path}')
    return model

# Predict using a saved model
def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print(f'Predicted Label: {predicted.item()}')
    return predicted.item()

if __name__ == '__main__':
    # # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Prepare data
    # train_loader, test_loader = prepare_data()

    # # Initialize model, loss function, and optimizer
    # model = CNN().to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Train the model
    # train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # # Test the model
    # test_model(model, test_loader, device)

    # # Save the model
    # save_model(model, './models/mnist_cnn.pth')

    # # Load the model and make predictions
    model = load_model('./models/mnist_cnn.pth', device)
    predict_image(model, './data/aaa.png', device)
