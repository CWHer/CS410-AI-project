import itertools
import pickle
from collections import deque, namedtuple
from itertools import product

import numpy as np
import torch
from agent.utils import Actions, ActionsFilter
from config import MDP_CONFIG, TRAIN_CONFIG
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset
from utils import plotHeatMaps, plotSparseMatrix


class ReplayBuffer():
    Data = namedtuple("data", "state mcts_prob value")

    def __init__(self) -> None:
        # NOTE: separate 10x20 and 20x10
        self.buffer = [
            deque(maxlen=TRAIN_CONFIG.replay_size)
            for _ in range(2)]

    def size(self):
        return len(self.buffer[0]) + len(self.buffer[1])

    def save(self, version="w"):
        dataset_dir = TRAIN_CONFIG.dataset_dir

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        print("save replay buffer version({})".format(version))
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_dir):
        print("load replay buffer {}".format(data_dir))
        with open(data_dir, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        """[summary]
        whether data is enough to start training
        """
        return self.size() > TRAIN_CONFIG.train_threshold

    def __enhanceData(self, states, mcts_probs, values):
        """[summary]
        enhance data by rotating and flipping
        """
        data = [[], []]
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            for i in range(4):
                # rotate
                new_state = np.rot90(state, i, axes=(1, 2))
                new_mcts_prob = np.zeros_like(mcts_prob)
                indices = list(
                    map(lambda x: ActionsFilter.rot90(x, i),
                        range(MDP_CONFIG.action_size)))
                new_mcts_prob[np.array(indices)] = mcts_prob
                data[i & 1].append((new_state, new_mcts_prob, value))

                # debug
                # plotHeatMaps(
                #     ic(new_mcts_prob.reshape((len(Actions), ) * 3)), Actions)
                # plotSparseMatrix(new_state[0], "none")

                # flip
                new_state = np.array([np.fliplr(s) for s in new_state])
                new_mcts_prob = np.zeros_like(mcts_prob)
                indices = list(
                    map(ActionsFilter.fliplr, indices))
                new_mcts_prob[np.array(indices)] = mcts_prob
                data[i & 1].append((new_state, new_mcts_prob, value))

                # debug
                # ic(np.array(indices).reshape((len(Actions), ) * 3))
                # plotHeatMaps(
                #     ic(new_mcts_prob.reshape((len(Actions), ) * 3)), Actions)
                # plotSparseMatrix(new_state[0], "none")

        return data

    def add(self, states, mcts_probs, values):
        data = self.__enhanceData(
            states, mcts_probs, values)
        for i in range(2):
            self.buffer[i].extend(data[i])

    def trainIter(self):
        """[summary]
        generate dataset iterator for training
        NOTE: return 2 iterators that are from different buffers
        """
        train_iter = [None] * 2
        for i in range(2):
            states, mcts_probs, values = map(
                torch.tensor, zip(*self.buffer[i]))
            data_set = TensorDataset(states, mcts_probs, values)
            train_iter[i] = DataLoader(
                data_set, TRAIN_CONFIG.batch_size, shuffle=True)
        return train_iter
