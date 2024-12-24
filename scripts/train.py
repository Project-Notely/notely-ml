import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from datetime import datetime, timedelta
from src.models.recognition import HandwrittingCNN
from src.utils.preprocess import EMNISTPreprocessor


def format_time(seconds: float) -> str:
    """convert seconds to human readable string"""
    return str(timedelta(seconds=int(seconds)))


# record start time
start_time: float = time.time()

# init the EMNIST data preprocessor and load the datasets
# this will download EMNIST if not already present and create data loaders
preprocessor = EMNISTPreprocessor()
train_loader, test_loader = preprocessor.load_datasets()

# set up the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init the CNN model and move it to the selected device
# num_classes=62 because EMNIST byclass has 62 different characters
model = HandwrittingCNN(num_classes=62).to(device)

# init the Adam optimizer with a learning rate of 0.001
# Adam is an adaptive learning rate optimization algorithm
optimizer = optim.Adam(model.parameters(), lr=0.001)

# init the loss function
# CrossEntropyLoss combines LogSoftmax and NLLLoss in a single function
# good for training a classification problem with multiple classes
criterion = nn.CrossEntropyLoss()


# training loop
def train(
    model: HandwrittingCNN,
    loader: DataLoader,
    optimizer: optim.Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """
    Training function that runs one epoch of training

    Args:
        model: Neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer algorithm
        criterion: Loss function
        device: Device to use for training (GPU/CPU)

    Returns:
        float: Average loss for this epoch
    """
    # set model to training mode (enables dropout, batch normalization, etc.)
    model.train()
    running_loss = 0.0

    # iterate over batches of data
    for images, labels in loader:
        # move data to the selected device
        images, labels = images.to(device), labels.to(device)

        # zero the gradients
        # this is necessary because gradients are accumulated in each batch
        optimizer.zero_grad()

        # forward pass
        # compute predicted putputs by passing images to the model
        outputs = model(images)

        # compute the loss between predicted and actual labels
        loss = criterion(outputs, labels)

        # backwards pass
        # compute the gradient of the loss with respect to the model parameters
        loss.backward()

        # update weights
        optimizer.step()

        # add the loss for this batch to the running loss
        running_loss += loss.item()

    # return the average loss for this epoch
    return running_loss / len(loader)


# testing loop
def test(model: HandwrittingCNN, loader: DataLoader, device: torch.device) -> float:
    """
    Testing function to evalulate model performance

    Args:
        model: Neural network model
        loader: DataLoader containing test data
        device: Device to use for testing (GPU/CPU)

    Returns:
        float: Accuracy of the model on the test data
    """
    # set model to evalulate mode (disables dropout, batch normalization, etc.)
    model.eval()
    correct = 0
    total = 0

    # disable gradient calculation for efficiency
    with torch.no_grad():
        for images, labels in loader:
            # move data to the selected device
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)

            # get predictions
            # torch.max returns (values, indices); we want indices
            _, predicted = torch.max(outputs.data, 1)

            # update statistics
            total += labels.size(0)  # add batch size
            correct += (
                (predicted == labels).sum().item()
            )  # add number of correct predictions

    # return the accuracy of the model on the test data
    return correct / total


# saving model
def save_checkpoint(
    model: HandwrittingCNN,
    optimizer: optim.Adam,
    epoch: int,
    accuracy: float,
    path: str = "checkpoint",
):
    """
    Save model checkpoint to disk

    Args:
        model: Neural network model
        optimizer: Optimizer algorithm
        epoch: Current epoch number
        accuracy: Accuracy of the model on the test data
        path: Path to save the checkpoint
    """
    import os

    # create checkpoint directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # save the checkpoint to disk
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
        },
        f"{path}/model_epoch_{epoch}.pth",
    )


# training loop with progress tracking
print(f"Using device: {device}")
print("Starting training...")

epochs = 10
best_accuracy = 0.0
epoch_times = []

# iterate through epochs
for epoch in range(epochs):
    epoch_start = time.time()

    # train one epoch and get average loss
    train_loss = train(model, train_loader, optimizer, criterion, device)

    # evalulate model on test set
    test_acc = test(model, test_loader, device)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Time taken: {format_time(epoch_time)}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        save_checkpoint(model, optimizer, epoch + 1, best_accuracy)
        print(f"Saved new best model with accuracy: {best_accuracy:.4f}")

    print("-" * 50)

# calculate final timing statistics
total_time = time.time() - start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)


print("Training complete!")
print(f"Best accuracy achieved: {best_accuracy:.4f}")
