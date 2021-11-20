import datetime
import json
import time
from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "board_height": 10,
    "board_width": 20,
    "board_size": 10 * 20,
    "action_size": 64,
    "c_reward": 0.1,
    "final_reward": 2,

    # NOTE: the positions of beats are random,
    #   to mix prior probs requires samples
    "state_sample_num": 4,
}

NETWORK_CONFIG = {
    "periods_num": 3,
    "num_channels": 128,
    "num_res": 4,
    "value_scale": 2,
}

MCTS_CONFIG = {
    "inv_temperature": 1/1,
    # "search_num": 1000,
    "search_num": 10,
    "gamma": 0.98,              # foresee 100 steps
    "c_puct": 10,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps": 0.1,
}

TRAIN_CONFIG = {
    "train_epochs": 5,
    "c_loss": 0.1,
    "l2_weight": 1e-4,
    "learning_rate": 0.001,
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
    # evaluate model
    "check_freq": 15,
    "update_threshold": 0.55,
    "num_contest": 20,
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

MCTS_CONFIG_TYPE = namedtuple("MCTS_CONFIG", MCTS_CONFIG.keys())
MCTS_CONFIG = MCTS_CONFIG_TYPE._make(MCTS_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
