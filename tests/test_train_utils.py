from itertools import chain, zip_longest

import torch
from icecream import ic
from tqdm import tqdm

from agent.network import D3QN
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer

torch.manual_seed(0)
net = D3QN()
# net.setDevice(torch.device("cpu"))
episode_data = selfPlay(net, epsilon=0.1, seed=17)

# construct dataset
replay_buf = ReplayBuffer()
replay_buf.add(episode_data)
replay_buf.save(version="test")

winner = contest(net, net, seed=2)
ic(winner)

# test training
ic(replay_buf.size())
train_iter = replay_buf.trainIter()
iter_len = len(train_iter[0]) + len(train_iter[1])
for i in range(2):
    with tqdm(total=iter_len) as pbar:
        for data_batch in chain(*zip(*train_iter)):
            loss = net.trainStep(data_batch)
            # print(loss)
            pbar.update()
