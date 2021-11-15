from sys import version
from agent.network import PolicyValueNet
from train_utils.game import selfPlay
from train_utils.replay_buffer import ReplayBuffer

net = PolicyValueNet()
states, mcts_probs, values = selfPlay(net)
# construct dataset
replay_buf = ReplayBuffer()
replay_buf.add(states, mcts_probs, values)
replay_buf.save(version="test")
train_iter = replay_buf.trainIter()
for data_batch in train_iter:
    net.trainStep(data_batch)
