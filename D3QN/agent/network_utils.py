import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1)


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.net = nn.Sequential(
            conv3x3(num_channels, num_channels),
            # nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            conv3x3(num_channels, num_channels),
            # nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        y = self.net(x)
        return F.relu(x + y)
