from icecream import ic
from agent.network import PolicyValueNet
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer

from tqdm import tqdm

net = PolicyValueNet()
states, mcts_probs, values = selfPlay(net, seed=1)

# construct dataset
replay_buf = ReplayBuffer()
replay_buf.add(states, mcts_probs, values)
replay_buf.save(version="test")

winner = contest(net, net, seed=1)
ic(winner)

# test training
ic(replay_buf.size())
train_iter, iter_len = replay_buf.trainIter()
ic(iter_len)
with tqdm(total=iter_len) as pbar:
    for data_batch in train_iter:
        loss, acc = net.trainStep(data_batch)
        ic(loss, acc)
        pbar.update()
