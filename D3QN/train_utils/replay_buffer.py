import itertools
import pickle
import random
from collections import deque, namedtuple
from itertools import product

import numpy as np
import torch
from config import MDP_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG
from env.utils import Actions, ActionsFilter
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset
from utils import plotHeatMaps, plotSparseMatrix, timeLog


class ReplayBuffer():
    Data = namedtuple("data", "state reward action next_state done")

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

    def __enhanceData(self,
                      states, rewards, actions,
                      prev_actions, next_states, dones):
        """[summary]
        enhance data by rotating and flipping
        """
        enhanced_data = [[], []]
        for state, reward, action, prev_action, next_state, done in zip(
                states, rewards, actions, prev_actions, next_states, dones):
            # rotating and flipping
            for i in TRAIN_CONFIG.rot90_arr:
                # rotate
                new_state = np.rot90(state, i, axes=(1, 2))
                new_next_state = np.rot90(next_state, i, axes=(1, 2))
                new_action = ActionsFilter.rot90(action, i)
                # new_prev_action = list(
                #     map(lambda x: Actions.rot90(x, i), prev_action))
                # new_actions = ActionsFilter.genActions(new_prev_action)
                # NOTE: calculate relative index
                enhanced_data[i & 1].append(
                    (new_state, reward,
                     new_action, new_next_state, done))

                if not TRAIN_CONFIG.enable_enhance:
                    break

                # debug
                # ic(new_prev_action)
                # ic(new_action)
                # print(ActionsFilter.Idx2Act(new_action))
                # print(new_actions.index(new_action))
                # plotSparseMatrix(new_state[2], "none")
                # plotSparseMatrix(new_next_state[2], "none")

                # flip
                new_state = np.array([np.fliplr(s) for s in new_state])
                new_next_state = np.array([
                    np.fliplr(s) for s in new_next_state])
                new_action = ActionsFilter.fliplr(new_action)
                # new_prev_action = list(
                #     map(lambda x: Actions.fliplr(x), new_prev_action))
                # new_actions = ActionsFilter.genActions(new_prev_action)
                # NOTE: calculate relative index
                enhanced_data[i & 1].append(
                    (new_state, reward,
                     new_action, new_next_state, done))

                # debug
                # ic(new_prev_action)
                # ic(new_action)
                # print(ActionsFilter.Idx2Act(new_action))
                # print(new_actions.index(new_action))
                # plotSparseMatrix(new_state[2], "none")
                # plotSparseMatrix(new_next_state[2], "none")

        # debug
        # for i in range(4):
        #     plotSparseMatrix(enhanced_data[0][i][0][2], "none")
        return enhanced_data

    def add(self, episode_data):
        enhanced_data = self.__enhanceData(*episode_data)
        for i in range(2):
            self.buffer[i].extend(enhanced_data[i])

    def sample(self):
        k = random.choice(
            [i for i in range(2)
             if len(self.buffer[i]) > TRAIN_CONFIG.batch_size])
        indices = np.random.choice(
            len(self.buffer[k]), TRAIN_CONFIG.batch_size)
        data_batch = map(
            lambda x: torch.from_numpy(np.stack(x)),
            zip(*[self.buffer[k][i] for i in indices])
        )
        return list(data_batch)

    @timeLog
    def trainIter(self):
        """[summary]
        generate dataset iterator for training
        NOTE: return 2 iterators that are from different buffers
        """
        train_iter = [None] * 2
        for i in range(2):
            # states, rewards, actions, next_states, dones
            data_set = list(map(
                lambda x: torch.from_numpy(np.stack(x)),
                zip(*self.buffer[i])))
            data_set = TensorDataset(*data_set)
            train_iter[i] = DataLoader(
                data_set, TRAIN_CONFIG.batch_size, shuffle=True)
        return train_iter
