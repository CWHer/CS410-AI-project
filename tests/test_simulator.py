import random

import numpy as np
import torch
from icecream import ic

from env.simulator import Simulator
from utils import plotSparseMatrix


def showFeatures(features):
    for i in range(features.shape[0]):
        plotSparseMatrix(features[i], "none")


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

env = Simulator()
features = env.reset()
# showFeatures(features[0])
# showFeatures(features[1])

while True:
    indices = env.validActions()
    joint_action = [
        random.choice(indices[0]), random.choice(indices[1])]
    features, reward, done, _ = env.step(joint_action)
    ic(reward)
    # showFeatures(features[0])
    # showFeatures(features[1])

    if done:
        break
