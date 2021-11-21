from itertools import chain

import torch
from icecream import ic
from tqdm import tqdm

from agent.network import PolicyValueNet
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer

net = PolicyValueNet()
# net.setDevice(torch.device("cpu"))
states, mcts_probs, values = selfPlay(net, seed=1)

# construct dataset
replay_buf = ReplayBuffer()
replay_buf.add(states, mcts_probs, values)
replay_buf.save(version="test")

winner = contest(net, net, seed=1)
ic(winner)

# test training
ic(replay_buf.size())
train_iter = replay_buf.trainIter()
iter_len = len(train_iter[0]) + len(train_iter[1])
for i in range(2):
    with tqdm(total=iter_len) as pbar:
        for data_batch in chain(*train_iter):
            loss, acc = net.trainStep(data_batch)
            ic(loss, acc)
            pbar.update()
