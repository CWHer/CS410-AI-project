import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import NETWORK_CONFIG, TRAIN_CONFIG
from icecream import ic

from .network_utils import ResBlock, conv3x3


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

    def save(self, version="w"):
        checkpoint_dir = TRAIN_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        torch.save(
            self.net.state_dict(),
            checkpoint_dir + f"/model_{version}")

    def load(self, model_dir):
        self.net.load_state_dict(torch.load(
            model_dir, map_location=torch.device("cuda:0")))

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
        policy_log = policy_log.squeeze(dim=0).cpu().detach()
        return (
            np.exp(policy_log.numpy()),
            value.item())

    def trainStep(self, data_batch: torch.Tensor):
        self.net.train()
        states, mcts_probs, values = data_batch

        # loss function: (z - v) ^ 2 - pi ^ T log(p) + c | theta | ^ 2
        policy_log, v = self.net(states.float().cuda())
        # TODO: calculate accuracy

        value_loss = F.mse_loss(v.view(-1), values.float().cuda())
        policy_loss = -torch.sum(
            policy_log * mcts_probs.float().cuda(), dim=1).mean()
        loss = value_loss * TRAIN_CONFIG.c_loss + policy_loss

        # debug: torchviz
        # from torchviz import make_dot
        # dot = make_dot(loss)
        # dot.format = "png"
        # dot.render("loss")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
