import copy
import random

import numpy as np
import torch
from icecream import ic

from env.snake_env.chooseenv import make

# FIX seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env = make("snakes_3v3", conf=None)
state = env.reset()

while True:
    joint_action = np.random.randint(4, size=6)
    action = env.encode(joint_action)
    env.draw_board()
    # NOTE: env supports deepcopy :)
    env_copy = copy.deepcopy(env)
    # NOTE: return values
    #   (all_observes, reward, done, info_before, info_after)
    next_state, reward, done, _, info = env.step(action)
    ic(next_state, info)
    next_state_copy, reward, done, _, info_copy = env_copy.step(action)
    ic(next_state_copy, info_copy)
    # break
