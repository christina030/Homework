import os

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block (used only in the Teacher network)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class TeacherNet(nn.Module):
    """ResNet-lite Teacher for 28x28 BloodMNIST images."""
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(24, 24, stride=1)   # 28x28
        self.layer2 = ResidualBlock(24, 48, stride=2)   # 14x14
        self.layer3 = ResidualBlock(48, 96, stride=2)   # 7x7
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class StudentNet(nn.Module):
    """Lightweight plain CNN Student (no residual connections, far fewer channels)."""
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model, path='/tmp/_tmp_model_size_probe.pt'):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb

