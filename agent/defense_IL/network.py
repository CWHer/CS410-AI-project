import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import MDP_CONFIG, NETWORK_CONFIG


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


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = NETWORK_CONFIG.in_channels
        hidden_channels = NETWORK_CONFIG.num_channels
        board_size = MDP_CONFIG.board_size

        resnets = [
            ResBlock(hidden_channels)
            for _ in range(NETWORK_CONFIG.num_res)]
        self.common_layers = nn.Sequential(
            conv3x3(in_channels, hidden_channels),
            nn.ReLU(), *resnets)

        # A head
        self.A_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(board_size * 4, board_size * 2),
            nn.ReLU(),
            nn.Linear(board_size * 2, MDP_CONFIG.action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.common_layers(x)
        logits = self.A_output(x)
        return logits


class Imitator():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")

        self.net = Network().to(self.device)

    def load(self, model_path):
        print("load network {}".format(model_path))

        self.net.load_state_dict(torch.load(
            model_path, map_location=self.device))

        # model = torch.load(
        #     model_path + "/Part1.pth", map_location=self.device)
        # model_rest = torch.load(
        #     model_path + "/Part2.pth", map_location=self.device)
        # model.update(model_rest)
        # self.net.load_state_dict(model)

    def predict(self, features):
        """[summary]
        NOTE: use encoder to encode state
            before calling predict
        """
        if features.ndim < 4:
            features = np.expand_dims(features, 0)

        features = torch.from_numpy(
            features).float().to(self.device)
        with torch.no_grad():
            self.net.eval()
            probs = self.net(features)
        return probs.detach().cpu().numpy()
