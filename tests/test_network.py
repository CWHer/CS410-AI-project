import torch
from icecream import ic
from torchsummary import summary

from agent.network import PolicyValueNet
from agent.network_utils import ObsEncoder
from env.chooseenv import make
from utils import plotSparseMatrix

net = PolicyValueNet()
summary(net.net, (14, 10, 20), batch_size=512)

env = make("snakes_3v3", conf=None)
state = env.reset()
# ic(state)
encoder = ObsEncoder()
encoder.add(state)
features = encoder.encode(turn=0)
# for i in range(features.shape[0]):
#     plotSparseMatrix(features[i], "none")
policy, value = net.predict(features)
ic(policy, value)
