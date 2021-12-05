import datetime
import json
from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "board_height": 10,
    "board_width": 20,
    "board_size": 10 * 20,
    "action_size": 4,
    "total_step": 200,
}

NETWORK_CONFIG = {
    "periods_num": 5,
    "in_channels":  None,
    "num_channels": 256,
    "num_res": 5,
}
NETWORK_CONFIG["in_channels"] = \
    NETWORK_CONFIG["periods_num"] * 2 + 6

TRAIN_CONFIG = {
    "score_threshold": 40,

    "train_epochs": 100,
    "patience": 10,
    "early_stop": 20,
    "l2_weight": 1e-4,
    "learning_rate": 1e-2,
    "checkpoint_dir": "checkpoint",
    "batch_size": 128,
    "dataset_dir": "dataset",
    "para_dir": "parameters",

    # evaluate model
    "process_num": 1,
    "check_freq": 20,
    "update_threshold": 0.55,
    "num_contest": 20,
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
