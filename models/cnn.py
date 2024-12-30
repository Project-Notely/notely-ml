import torch.nn as nn
import torch.nn.functional as F


class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=47):  # EMNIST balanced has 47 classes
        super(EMNISTCNN, self).__init__()

        self.features = nn.Sequential(
            # first convolution layer
            # input: 1 channel (grayscale image)
            # output: 32 channels (32 different feature maps)
            # kernel_size=3: 3x3 sliding window
            # padding=1: add 1 pixel border to preserve size
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # batch normalization
            nn.BatchNorm2d(32),
            # activation function
            nn.ReLU(inplace=True),
            # pooling layer
            # reduces the spatial dimensions by half
            # takes maximum value in each 2x2 window
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dropout layer
            # randomly zeroes some of the elements of the input tensor
            # with probability p using samples from a Bernoulli distribution
            nn.Dropout2d(0.25),
            # second convolution layer
            # input: 32 channels (from conv1)
            # output: 64 channels (64 different feature maps)
            # kernel_size=3: 3x3 sliding window
            # padding=1: add 1 pixel border to preserve size
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # batch normalization
            nn.BatchNorm2d(64),
            # activation function
            nn.ReLU(inplace=True),
            # pooling layer
            # reduces the spatial dimensions by half
            # takes maximum value in each 2x2 window
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dropout layer
            # randomly zeroes some of the elements of the input tensor
            # with probability p using samples from a Bernoulli distribution
            nn.Dropout2d(0.25),
            # third convolution layer
            # input: 64 channels (from conv2)
            # output: 128 channels (128 different feature maps)
            # kernel_size=3: 3x3 sliding window
            # padding=1: add 1 pixel border to preserve size
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # batch normalization
            nn.BatchNorm2d(128),
            # activation function
            nn.ReLU(inplace=True),
            # pooling layer
            # reduces the spatial dimensions by half
            # takes maximum value in each 2x2 window
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dropout layer
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            # first fully connected layer
            # input: 128 channels (from conv3)
            # 3 * 3 is the size of the output of the last convolutional layer
            # output: 512 neurons (512 different feature maps)
            nn.Linear(128 * 3 * 3, 512),
            # batch normalization
            nn.BatchNorm1d(512),
            # activation function
            nn.ReLU(inplace=True),
            # dropout layer
            # randomly zeroes some of the elements of the input tensor
            # with probability p using samples from a Bernoulli distribution
            nn.Dropout(0.5),
            # output layer
            # input: 512 neurons (from fc1)
            # output: num_classes neurons
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # pass the input through the convolutional layers
        x = self.features(x)
        # flatten the output for fully connected layers
        # x.size(0) is the batch size
        # -1 means the remaining dimensions are flattened
        x = x.view(x.size(0), -1)
        # pass the flattened output through the fully connected layers
        x = self.classifier(x)
        return x