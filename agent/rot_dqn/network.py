import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MDP_CONFIG, NETWORK_CONFIG


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


class VANet(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = NETWORK_CONFIG.in_channels
        hidden_channels = NETWORK_CONFIG.num_channels
        input_size = MDP_CONFIG.input_size

        self.common_layers = nn.Sequential(
            conv3x3(in_channels, hidden_channels),
            nn.ReLU(),
            *[ResBlock(hidden_channels)
              for _ in range(NETWORK_CONFIG.num_res)])

        # A head
        self.A_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * input_size, 2 * input_size),
            nn.ReLU(),
            nn.Linear(2 * input_size, MDP_CONFIG.action_size)
        )

        # V head
        self.V_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * input_size, 2 * input_size),
            nn.ReLU(),
            nn.Linear(2 * input_size, 1)
        )

    def forward(self, x):
        x = self.common_layers(x)
        A = self.A_output(x)
        V = self.V_output(x)
        Q = V + A - A.mean(dim=1).view(-1, 1)
        return Q


class D3QN():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")

        self.q_net = VANet().to(self.device)
        # NOTE: target net is used for training
        self.target_net = VANet().to(self.device)

    def load(self, model_dir):
        print("load network {}".format(model_dir))

        self.q_net.load_state_dict(torch.load(
            model_dir, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

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
            q_values = self.q_net(features)
        return q_values.detach().cpu().numpy()
