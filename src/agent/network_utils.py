import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MDP_CONFIG, NETWORK_CONFIG
from icecream import ic

from .utils import ActionsFilter


class ObsEncoder():
    def __init__(self) -> None:
        self.states = deque(
            maxlen=NETWORK_CONFIG.periods_num)

    def getLastActions(self):
        """[summary]
        get actions of last state
        """
        return ActionsFilter.extractActions(self.states[-1][1:])

    def add(self, raw_state):
        """[summary]
        add (fill) new state to deque 
        NOTE: new state is repeated to FILL deque

        Args:
            raw_state ([type]): [description] output of env
        """
        raw_state = copy.deepcopy(raw_state[0])
        state = [raw_state[i] for i in range(1, 8)]
        if self.states:
            self.states.popleft()
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
            (periods_num * 2 + 6,
             MDP_CONFIG.board_height,
             MDP_CONFIG.board_width))

        # positions of my body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in my_indices:
                features[i][tuple(zip(*self.states[i][j]))] = 1
        # current positions of my head
        for i, idx in enumerate(my_indices):
            x, y = self.states[-1][idx][0]
            features[periods_num + i][x][y] = 1

        # positions of opponent's body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in opponent_indices:
                features[periods_num + i +
                         3][tuple(zip(*self.states[i][j]))] = 1
        # current positions of opponent's head
        for i in opponent_indices:
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 3][x][y] = 1

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

        self.net = nn.Sequential(
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        y = self.net(x)
        return F.relu(x + y)