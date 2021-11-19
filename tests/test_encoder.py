import random

import numpy as np
import torch
from icecream import ic
from torchsummary import summary

from agent.network import PolicyValueNet
from agent.network_utils import ObsEncoder
from env.chooseenv import make
from utils import plotSparseMatrix

# FIX seed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

encoder = ObsEncoder()
net = PolicyValueNet()
summary(net.net, (10, 10, 20), batch_size=512)

env = make("snakes_3v3", conf=None)
state = env.reset()
encoder.add(state)
env.draw_board()

for i in range(2):
    joint_action = np.random.randint(4, size=6)
    action = env.encode(joint_action)
    next_state, *_ = env.step(action)
    encoder.add(next_state)
    ic(encoder.getLastActions())
    env.draw_board()

features = encoder.encode(turn=0)
for i in range(features.shape[0]):
    plotSparseMatrix(features[i], "none")

policy, value = net.predict(features)
ic(policy, value)
