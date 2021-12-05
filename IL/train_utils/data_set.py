import pickle

import numpy as np
import torch
from config import TRAIN_CONFIG
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset, dataset
from tqdm import tqdm
from utils import timeLog


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

        return state, [actions[:3], actions[3:]]

    @staticmethod
    def construct(task_path, version="xxx"):
        from .encoder import ObsEncoder

        with open(task_path, "rb") as f:
            task_buffer = pickle.load(f)

        data_buffer = []
        for task in tqdm(task_buffer):
            # NOTE: only collect data that yields
            #   total score > score_threshold
            collect_options = [
                sum(task["n_return"][:3]) > TRAIN_CONFIG.score_threshold,
                sum(task["n_return"][3:]) > TRAIN_CONFIG.score_threshold]
            if not any(collect_options):
                continue

            encoder = ObsEncoder()
            state, _ = DataSet.decodeLog(task["init_info"])
            encoder.add([state])
            for i, step in enumerate(task["steps"]):
                state, actions = DataSet.decodeLog(step["info_after"])
                for k in range(2):
                    if collect_options[k]:
                        data_buffer.append(
                            (encoder.encode(turn=k, num_step=i), actions[k]))
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
    def trainIter(self):
        data_set = list(map(
            lambda x: torch.from_numpy(np.stack(x)),
            zip(*self.data_buffer)))
        data_set = TensorDataset(*data_set)
        train_iter = DataLoader(
            data_set, TRAIN_CONFIG.batch_size)
        return train_iter
