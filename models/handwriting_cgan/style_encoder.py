import torch.nn as nn
import torch


class StyleEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        super(StyleEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 8 * 8, feature_dim)  # assumes 64x64 input size

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        latent_style = self.fc(features)
        return latent_style
