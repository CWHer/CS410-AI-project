import random

import numpy as np
import torch
from icecream import ic
from torchsummary import summary

from agent.network import D3QN
from config import NETWORK_CONFIG
from env.simulator import Simulator
from utils import plotSparseMatrix


def showFeatures(features):
    for i in range(features.shape[0]):
        plotSparseMatrix(features[i], "none")


# FIX seed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

net = D3QN()
summary(net.q_net, (NETWORK_CONFIG.in_channels, 10, 20), batch_size=512)

env = Simulator()
features = env.reset()
env.drawBoard()

for i in range(10):
    indices = env.validActions()
    joint_action = [None] * 6
    for k in range(6):
        q_values = net.predict(features[k])
        action = q_values.argmax(axis=1).item()
        # ic(action, q_values.max())
        joint_action[k] = indices[k][action]

    next_features, reward, done, _ = env.step(joint_action)
    features = next_features

    env.drawBoard()

showFeatures(next_features[0])
ic()
