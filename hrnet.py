import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out)

class HRNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(HRNet, self).__init__()
        self.layer1 = self._make_layer(BasicBlock, in_channels, 64, 4)
        self.transition1 = self._make_transition_layer(64, 128)
        self.layer2 = self._make_layer(BasicBlock, 128, 128, 4)
        self.transition2 = self._make_transition_layer(128, 256)
        self.layer3 = self._make_layer(BasicBlock, 256, 256, 4)
        self.transition3 = self._make_transition_layer(256, 512)
        self.layer4 = self._make_layer(BasicBlock, 512, 512, 4)
        self.final_layer = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        x = self.final_layer(x)
        return torch.sigmoid(x)