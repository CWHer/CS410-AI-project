import pickle

import numpy as np
import torch
from config import TRAIN_CONFIG
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import timeLog
from itertools import filterfalse


class DataSet():
    def __init__(self) -> None:
        # (s, a): np.ndarray
        self.data_buffer = []

    @staticmethod
    def decodeLog(data):
        state = {i: [] for i in range(1, 8)}
        state[1] = data["beans_position"]
        for i in range(6):
            state[i + 2] = data["snakes_position"][i]

        directions = {"up": 0, "down": 1, "left": 2, "right": 3}
        actions = list(map(lambda x: directions[x], data["directions"]))

        return state, actions

    @staticmethod
    def construct(task_path, version="xxx"):
        from .encoder import ObsEncoder

        with open(task_path, "rb") as f:
            task_buffer = pickle.load(f)

        data_buffer = []
        indices = [[1, 2, 3], [4, 5, 6]]
        for task in tqdm(task_buffer):
            score_threshold = TRAIN_CONFIG.score_threshold
            # NOTE: only collect data that yields
            #   total score > score_threshold
            collect_options = [
                sum(task["n_return"][:3]) > score_threshold,
                sum(task["n_return"][3:]) > score_threshold]
            if not any(collect_options):
                continue

            encoder = ObsEncoder()
            state, _ = DataSet.decodeLog(task["init_info"])
            encoder.add([state])
            for t, step in enumerate(task["steps"]):
                state, actions = DataSet.decodeLog(step["info_after"])
                for i in filterfalse(
                        lambda x: not collect_options[x], range(2)):
                    for index in indices[i]:
                        data_buffer.append(
                            (encoder.encode(idx=index, num_step=t),
                                actions[index - 1]))
                encoder.add([state])

        ic(len(data_buffer))

        import os
        dataset_dir = TRAIN_CONFIG.dataset_dir
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        print("save data {} from {}".format(version, task_path))
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(data_buffer, f)

    def load(self, data_path):
        print("load data set {}".format(data_path))
        with open(data_path, "rb") as f:
            self.data_buffer = pickle.load(f)

    def __enhanceData(self):
        # TODO
        raise NotImplementedError()

    @timeLog
    def getIter(self):
        data_num = int(len(self.data_buffer) * 0.7)

        train_set = list(map(
            lambda x: torch.from_numpy(np.stack(x)),
            zip(*self.data_buffer[:data_num])))
        train_set = TensorDataset(*train_set)
        train_iter = DataLoader(
            train_set, TRAIN_CONFIG.batch_size, shuffle=True)

        valid_set = list(map(
            lambda x: torch.from_numpy(np.stack(x)),
            zip(*self.data_buffer[data_num:])))
        valid_set = TensorDataset(*valid_set)
        valid_iter = DataLoader(
            valid_set, TRAIN_CONFIG.batch_size, shuffle=True)

        return train_iter, valid_iter
