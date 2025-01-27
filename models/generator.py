import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, style_dim, content_dim, output_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_dim + content_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, style_features, content_features):
        combined = torch.cat([style_features, content_features], dim=1)
        x = self.fc(combined)
        x = x.view(x.size(0), 256, 8, 8)  # reshape to feature map
        img = self.deconv(x)
        return img
