import torch
import torch.nn as nn


class HandwrittingRecognitionModel(nn.Module):
    def __init__(self, num_classes: int):
        super(HandwrittingRecognitionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.rnn = nn.LSTM(
            128 * 8, 256, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, 2, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
