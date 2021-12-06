

# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""

import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

# fmt: off
from DQN import DQN
from feature import *
# fmt: on

model = DQN(state_dim=100, action_dim=3, isTrain=False)
file_path = os.path.dirname(
    os.path.abspath(__file__)) + "/params.pth"
model.load(file_path)

head_action = {"up": [2, 1, 3], "right": [
    1, 3, 0], "down": [3, 0, 2], "left": [0, 2, 1]}


def output_to_action(output, direction):
    transf = head_action[direction]
    return transf[output]


def my_controller(observation, action_space, is_act_continuous=False):
    # 2, 3, 4, 5, 6, 7
    agent = observation['controlled_snake_index']
    direction, feature = head_and_obs(observation, agent)
    out = model.egreedy_action(np.array(feature))
    action = output_to_action(out, direction)
    action = again_action(observation, action)
    action_mat = [[0, 0, 0, 0]]
    action_mat[0][action] = 1
    return action_mat
