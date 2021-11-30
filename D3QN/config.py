import datetime
import json
from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "board_height": 10,
    "board_width": 20,
    "board_size": 10 * 20,
    "action_size": 27,
    "c_reward": 0.2,
    "final_reward": 10,
    "gamma": 0.98,
    "total_step": 200,
}

NETWORK_CONFIG = {
    "periods_num": 5,
    "in_channels":  None,
    "num_channels": 128,
    "num_res": 2,
    "update_freq": 100,
}
NETWORK_CONFIG["in_channels"] = \
    NETWORK_CONFIG["periods_num"] * 2 + 6

TRAIN_CONFIG = {
    "train_num": 100000,
    "train_epochs": 2000,
    # "train_epochs": 5,
    "learning_rate": 1e-4,
    "checkpoint_dir": "checkpoint",
    "batch_size": 256,
    "train_threshold": 10000,
    "replay_size": 1000000,
    "dataset_dir": "dataset",
    "data_save_freq": 500,
    "para_dir": "parameters",

    "process_num": 5,
    # evaluate model
    "check_freq": 10,
    "update_threshold": 0.55,
    "num_contest": 20,
    # data generation
    "game_num": 10,

    # epsilon-greedy
    "init_epsilon": 0.8,
    "min_epsilon": 0.1,
    "delta_epsilon": 0.01
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
            [MDP_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG],
            f, indent=4)


MDP_CONFIG_TYPE = namedtuple("MDP_CONFIG", MDP_CONFIG.keys())
MDP_CONFIG = MDP_CONFIG_TYPE._make(MDP_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
