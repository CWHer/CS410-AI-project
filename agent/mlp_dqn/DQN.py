import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import random

BATCH_SIZE = 256
GAMMA = 0.99
INITIAL_EPSILON = 1
DECAY_RATE = 1
REPLAY_SIZE = 50000
TAU = 0.02
TARGET_NETWORK_REPLACE_FREQ = 100  # 网络更新频率
np.random.seed(2)
torch.manual_seed(2)
random.seed(2)


class NET(nn.Module):  # 神经网络state_dim->512->256->128->action_dim
    def __init__(self, state_dim, action_dim):
        super(NET, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.lin3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.lin4 = nn.Linear(128, action_dim)

    def forward(self, state):
        # state = state.view(-1, len(state))
        state = state.to(torch.float32)
        feature = self.lin1(state)
        feature = self.lin2(feature)
        feature = self.lin3(feature)
        out_feature = self.lin4(feature)
        return out_feature


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = []
        self.max_size = buffer_size
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        transition_tuple = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition_tuple)

    def get_batches(self):
        sample_batch = random.sample(self.buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in sample_batch])
        action_batches = np.array([_[1] for _ in sample_batch])
        reward_batches = np.array([_[2] for _ in sample_batch])
        next_state_batches = np.array([_[3] for _ in sample_batch])
        done = np.array([_[4] for _ in sample_batch])

        return state_batches, action_batches, reward_batches, next_state_batches, done

    def __len__(self):
        return len(self.buffer)


class DQN(object):
    def __init__(self, state_dim, action_dim, isTrain):
        self.device = torch.device("cpu")
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE, BATCH_SIZE)
        # self.replay_buffer = ReplayBuffer(REPLAY_SIZE, BATCH_SIZE)
        self.time_step = 0
        self.tau = TAU
        self.epsilon = INITIAL_EPSILON
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.IsTrain = isTrain
        # 定义网络，损失函数，优化器
        self.eval_net, self.target_net = NET(
            self.state_dim, self.action_dim), NET(self.state_dim,  self.action_dim)
        self.eval_net, self.target_net = self.eval_net.to(
            self.device), self.target_net.to(self.device)
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=5e-5)
        self.LOSS = 0

    def egreedy_action(self, state):
        state = torch.from_numpy(
            state).view(-1, self.state_dim).to(self.device)
        Q_next = self.target_net(state).detach()
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon - 0.00004
        else:
            self.epsilon *= DECAY_RATE
        if self.IsTrain and random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            Q_next = Q_next.cpu()  # 将数据由gpu转向cpu
            return np.argmax(Q_next.numpy())

    def action(self, state):
        state = torch.from_numpy(state).to(self.device)
        Q_next = self.target_net(state).detach().cpu().numpy()
        return np.argmax(Q_next)

    def train(self):
        self.time_step += 1
        state_batch, action_batch, reward_batch, next_state_batch, done = self.replay_buffer.get_batches()
        state_batch = torch.tensor(state_batch).to(self.device)
        action_batch = torch.tensor(action_batch).view(
            BATCH_SIZE, 1).to(self.device)  # 转换成batch*1的tensor
        reward_batch = torch.tensor(reward_batch).view(
            BATCH_SIZE, 1).to(self.device)  # 转换成batch*1的tensor
        next_state_batch = torch.tensor(next_state_batch).to(self.device)
        done = torch.tensor(done).view(BATCH_SIZE, 1).to(self.device)
        # print(done)

        Q_eval = self.eval_net(state_batch).gather(
            1, action_batch)  # (batch_size, 1), eval中动作a对应的Q值
        Q_next = self.target_net(next_state_batch).detach()  # 下一个状态的Q值，并且不反向传播
        Q_target = reward_batch + \
            (1-done) * GAMMA * \
            Q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)，Q的近似值

        loss = self.loss_fun(Q_eval, Q_target)
        self.LOSS += loss.item()
        if (self.time_step + 1) % 10000 == 0:
            print(self.time_step + 1, "loss:", self.LOSS / 10000)
            self.LOSS = 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def load(self, name):
        self.target_net.load_state_dict(
            torch.load(name, map_location=self.device))
        self.eval_net.load_state_dict(
            self.target_net.state_dict())

    def save_paramaters(self, name):
        torch.save(self.target_net.state_dict(), name)
        print("victor:", name)
