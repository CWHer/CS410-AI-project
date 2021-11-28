import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import MDP_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG
from icecream import ic

from .network_utils import ResBlock, conv3x3


class VANet(nn.Module):
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
            nn.Linear(4 * board_size, MDP_CONFIG.action_size),
        )

        # V head
        self.V_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
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
        self.updateTarget()
        self.update_cnt = 0

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=TRAIN_CONFIG.learning_rate)

    def setDevice(self, device):
        self.device = device
        self.q_net.to(device)
        self.target_net.to(device)

    def save(self, version="D3QN"):
        checkpoint_dir = TRAIN_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        print("save network & optimizer / version({})".format(version))
        torch.save(
            self.target_net.state_dict(),
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_dir, optimizer_dir=None):
        print("load network {}".format(model_dir))

        self.q_net.load_state_dict(torch.load(
            model_dir, map_location=self.device))
        self.updateTarget()

        if optimizer_dir is None:
            print("load optimizer {}".format(optimizer_dir))
            self.optimizer.load_state_dict(torch.load(optimizer_dir))

    def updateTarget(self):
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

    def trainStep(self, data_batch):
        """[summary]

        Returns:
            loss
        """
        states, rewards, actions, next_states, dones = data_batch
        states = states.float().to(self.device)
        rewards = rewards.view(-1, 1).float().to(self.device)
        actions = actions.view(-1, 1).long().to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.view(-1, 1).float().to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_actions = self.q_net(next_states).argmax(dim=1)
            tq_values = self.target_net(
                next_states).gather(1, max_actions.view(-1, 1))

        q_targets = rewards + \
            (1 - dones) * MDP_CONFIG.gamma * tq_values
        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        if self.update_cnt == \
                NETWORK_CONFIG.update_freq:
            self.updateTarget()
            self.update_cnt = 0

        return loss.item()
