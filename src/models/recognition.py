import torch.nn as nn
import torch.nn.functional as F

class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=62):  # EMNIST byclass has 62 classes
        super(EMNISTCNN, self).__init__()
        
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
        # input: 1 channel (grayscale image)
        # output: 32 channels (32 different feature maps)
        # kernel_size=3: 3x3 sliding window
        # stride=1: move window 1 pixel at a time
        # padding=1: add 1 pixel border to preserve size
        
        # first pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # reduces the spatial dimensions by half
        # takes maximum value in each 2x2 window
        
        # second convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 64 filters
        # input: 32 channels (from conv1)
        # output: 64 channels (64 different feature maps)
        # kernel_size=3: 3x3 sliding window
        # stride=1: move window 1 pixel at a time
        # padding=1: add 1 pixel border to preserve size
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # second pooling layer
        # reduces the spatial dimension by half
        # takes maximum value in each 2x2 window
        
        # first fully connected layer
        # 64 * 7 * 7 is the output size of the last convolutional layer
        # output of conv1 is size 32 * 28 * 28
        # after pool1, the size is 32 * 14 * 14
        # output of conv2 is size 64 * 14 * 14
        # after pool2, the size is 64 * 7 * 7
        # 128 is the number of neurons in the first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # fully connected layer
        # input: flattened feature maps (64 channels * 7 * 7 pixels)
        # output: 128 neurons
        # this layer learns high-level combinations of features
        
        # output layer
        self.fc2 = nn.Linear(128, num_classes)  # output layer
        # input: 128 features from fc1
        # output: num_classes neurons (62 for EMNIST byclass)
        # each output corresponds to a different character class
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatten the output for fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # flatten the full connected layer
        # reshapes the 3D tensor to 2D
        # -1 means batch size is inferred
        # 64 * 7 * 7 is the number of features
        
        # first fully connected layer + relu
        x = F.relu(self.fc1(x))
        # apply fc1 and relu activation
        
        # output layer
        x = self.fc2(x)
        # final classification layer
        # output will be scores for each class
        
        return x
