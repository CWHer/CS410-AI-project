from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import NETWORK_CONFIG, TRAIN_CONFIG
from icecream import ic

# TODO
# HACK: padding pattern in Con2D need changes


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


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = NETWORK_CONFIG.periods_num * 2 + 4
        hidden_channels = NETWORK_CONFIG.num_channels
        board_size = NETWORK_CONFIG.board_size

        resnets = [
            ResBlock(hidden_channels)
            for _ in range(NETWORK_CONFIG.num_res)]
        self.common_layers = nn.Sequential(
            conv3x3(in_channels, hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(), *resnets)

        # policy head
        self.policy_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * board_size, NETWORK_CONFIG.action_size),
            nn.LogSoftmax(dim=1)
        )

        # value head
        self.value_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        # TODO: modify value_output

    def forward(self, x):
        x = self.common_layers(x)
        policy_log = self.policy_output(x)
        value = self.value_output(x)
        return policy_log, value


class PolicyValueNet():
    def __init__(self) -> None:
        self.net = Network().cuda()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=TRAIN_CONFIG.learning_rate,
            weight_decay=TRAIN_CONFIG.l2_weight)

    def save(self):
        pass
        # TODO

    def load(self, model_dir):
        pass
        # TODO

    def predict(self, features):
        """[summary]
        NOTE: use encoder to encode state 
            before calling predict

        """
        self.net.eval()
        features = torch.from_numpy(
            np.expand_dims(features, 0)).float().cuda()
        with torch.no_grad():
            policy_log, value = self.net(features)
        return (
            np.exp(policy_log.cpu().detach().numpy()),
            value.cpu().detach().numpy())

    def trainStep(self):
        self.net.train()
        # TODO
