from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NETWORK_CONFIG
from icecream import ic


class ObsEncoder():
    def __init__(self) -> None:
        self.states = deque(
            maxlen=NETWORK_CONFIG.periods_num)

    def add(self, raw_state):
        """[summary]

        Args:
            raw_state ([type]): [description] output of env
        """
        raw_state = raw_state[0]
        state = [raw_state[i] for i in range(1, 8)]
        while (len(self.states) <
               NETWORK_CONFIG.periods_num):
            self.states.append(state)

    def encode(self, turn) -> np.ndarray:
        """[summary]
        features selection

        """
        bean_index = 0
        my_indices = [1, 2, 3] if turn == 0 else [4, 5, 6]
        opponent_indices = [4, 5, 6] if turn == 0 else [1, 2, 3]

        periods_num = NETWORK_CONFIG.periods_num
        features = np.zeros(
            (periods_num * 2 + 4, 10, 20))

        # positions of my body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in my_indices:
                features[i][tuple(zip(*self.states[i][j]))] = 1
        # current positions of my head
        for i in my_indices:
            x, y = self.states[-1][i][0]
            features[periods_num][x][y] = 1

        # positions of opponent's body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in opponent_indices:
                features[periods_num + 1 +
                         i][tuple(zip(*self.states[i][j]))] = 1
        # current positions of opponent's head
        for i in opponent_indices:
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 1][x][y] = 1

        # positions of all the beans
        features[-2][tuple(zip(*self.states[-1][bean_index]))] = 1

        # all 0 if turn == 0 else all 1
        if turn == 1:
            features[-1, ...] = np.ones_like(features[-1, ...])

        return features


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, padding=1,
        padding_mode="circular", bias=False)


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = conv3x3(num_channels, num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)
