from collections import namedtuple, deque
import pickle
from config import TRAIN_CONFIG
import torch
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer():
    def __init__(self) -> None:
        self.Data = namedtuple("data", "state mcts_prob value")
        self.buffer = deque(maxlen=TRAIN_CONFIG.replay_size)

    def save(self, version="w"):
        dataset_dir = TRAIN_CONFIG.dataset_dir

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_dir):
        with open(data_dir, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        """[summary]
        whether data is enough to start training
        """
        return len(self.buffer) > TRAIN_CONFIG.train_threshold

    def __enhanceData(self, states, mcts_probs, values):
        pass
        # TODO: rotation & reflection
        return zip(states, mcts_probs, values)

    def add(self, states, mcts_probs, values):
        """[summary]

        """
        self.buffer.extend(
            self.__enhanceData(states, mcts_probs, values))

    def trainIter(self):
        """[summary]
        generate dataset iterator for training
        """
        states, mcts_probs, values = map(torch.tensor, zip(*self.buffer))
        data_set = TensorDataset(states, mcts_probs, values)
        train_iter = DataLoader(
            data_set, TRAIN_CONFIG.batch_size, shuffle=True)
        return train_iter
