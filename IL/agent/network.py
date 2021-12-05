import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import MDP_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG
from icecream import ic
from utils import printError

from .network_utils import ResBlock, conv3x3


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

        self.optimizer = optim.Adam(
            self.net.parameters(),
            weight_decay=TRAIN_CONFIG.l2_weight,
            lr=TRAIN_CONFIG.learning_rate)

    def setDevice(self, device):
        self.device = device
        self.net.to(device)

    def save(self, version="imitator"):
        checkpoint_dir = TRAIN_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        print("save network & optimizer / version({})".format(version))
        torch.save(
            self.net.state_dict(),
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_path, optimizer_path=None):
        print("load network {}".format(model_path))

        self.q_net.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

        if not optimizer_path is None:
            print("load optimizer {}".format(optimizer_path))
            self.optimizer.load_state_dict(torch.load(optimizer_path))

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
            logits = self.net(features)
        return logits.detach().cpu().numpy()

    def trainStep(self, data_batch, is_train):
        """[summary]

        Returns:
            loss
        """
        self.net.train()
        if not is_train:
            self.net.eval()

        states, expert_actions = data_batch
        states = states.float().to(self.device)
        expert_actions = expert_actions.long().to(self.device)

        logits = self.net(states)
        actions = logits.argmax(dim=1).view(-1, 1)
        expert_actions = expert_actions.view(-1, 1)

        accuracy = (
            actions == expert_actions).float().mean().item()
        loss = -torch.log(
            logits.gather(1, expert_actions)).mean()

        printError(torch.isnan(loss), "loss is nan!")

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), accuracy
