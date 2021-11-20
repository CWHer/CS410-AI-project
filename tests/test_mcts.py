import copy
import random

import numpy as np
import torch
from icecream import ic

from agent.mcts import MCTSPlayer
from agent.network import PolicyValueNet
from agent.network_utils import ObsEncoder
from agent.utils import ActionsFilter, ScoreBoard
from env.chooseenv import make
from utils import plotSparseMatrix

# FIX seeds
np.random.seed(2)
random.seed(2)
torch.manual_seed(2)

env = make("snakes_3v3", conf=None)
state = env.reset()
# env.draw_board()

net = PolicyValueNet()
net.setDevice(torch.device("cuda:0"))
# player = MCTSPlayer(net, 0)
# player = MCTSPlayer(net, 1)

encoder = ObsEncoder()
encoder.add(state)

score_board = ScoreBoard()

# case 1: RE test
# actions, mcts_prob = player.getAction(
#     env, encoder, score_board, is_train=True)

# case 2: iter = 37     (eat bean)
# case 3: iter = 199    (final)
for i in range(36):
    joint_action = np.random.randint(4, size=6)
    action = env.encode(joint_action)
    next_state, reward, done, _, info = env.step(action)
    encoder.add(next_state)
    score_board.add(reward)

# ic(env.beans_position)
joint_action = np.random.randint(4, size=6)
env.beans_position = [[5, 2], [0, 15], [0, 8], [7, 17], [6, 0]]
next_state, reward, done, _, info = env.step(action)
encoder.add(next_state)
score_board.add(reward)
ic(encoder.getLastActions())
env.draw_board()

ic(score_board.score0, score_board.score1)
player = MCTSPlayer(net, 0)

# case 2 / 3
action, mcts_prob = player.getAction(
    env, encoder, score_board, is_train=False)

ic(ActionsFilter.Idx2Act(action))
ic(done, score_board.getWinner())
