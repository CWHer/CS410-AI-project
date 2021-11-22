import datetime
import json
from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "board_height": 10,
    "board_width": 20,
    "board_size": 10 * 20,
    "action_size": 27,
    "c_reward": 1,
    "final_reward": 0,
    "gamma": 0.98,
}

NETWORK_CONFIG = {
    "periods_num": 3,
    "num_channels": 128,
    "num_res": 4,
}

TRAIN_CONFIG = {
    # TODO
    "train_epochs": 5,
    "c_loss": 0.1,
    "learning_rate": 5e-5,
    "checkpoint_dir": "checkpoint",
    "batch_size": 512,
    "train_threshold": 20000,
    "replay_size": 1000000,
    "dataset_dir": "dataset",
    "data_save_freq": 5,
    "para_dir": "parameters",

    # total number
    "train_num": 10000,
    "process_num": 1,
    # data generation
    "game_num": 1,
}


def saveSettings():
    """[summary]
    save parameters
    """
    para_dir = TRAIN_CONFIG.para_dir

    import os
    if not os.path.exists(para_dir):
        os.mkdir(para_dir)

    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    with open(para_dir +
              f"/para_{timestamp}.json", "w") as f:
        json.dump(
            [MDP_CONFIG, NETWORK_CONFIG,
             MCTS_CONFIG, TRAIN_CONFIG], f, indent=4)


MDP_CONFIG_TYPE = namedtuple("MDP_CONFIG", MDP_CONFIG.keys())
MDP_CONFIG = MDP_CONFIG_TYPE._make(MDP_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
