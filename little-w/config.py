from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "xxx": 1,
    "final_reward": 10,
}


NETWORK_CONFIG = {
    "periods_num": 5,
    "num_channels": 256,
    "num_res": 1,
    "board_size": 10 * 20,
    "action_size": 64,
}

MCTS_CONFIG = {
    "inv_temperature": 1/0.15,
    "search_num": 10000,
    "gamma": 0.9,
    "c_puct": 4
}

DATA_GEN_CONFIG = {

}

TRAIN_CONFIG = {
    "c_loss": 0.1,
    "l2_weight": 1e-4,
    "learning_rate": 0.001,
    "checkpoint_dir": "checkpoint",
    "batch_size": 512,
    "train_threshold": 10000,
    "replay_size": 1000000,
    "dataset_dir": "dataset",

}

MDP_CONFIG_TYPE = namedtuple("MDP_CONFIG", MDP_CONFIG.keys())
MDP_CONFIG = MDP_CONFIG_TYPE._make(MDP_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())

MCTS_CONFIG_TYPE = namedtuple("MCTS_CONFIG", MCTS_CONFIG.keys())
MCTS_CONFIG = MCTS_CONFIG_TYPE._make(MCTS_CONFIG.values())

DATA_GEN_CONFIG_TYPE = namedtuple("DATA_GEN_CONFIG", DATA_GEN_CONFIG.keys())
DATA_GEN_CONFIG = DATA_GEN_CONFIG_TYPE._make(DATA_GEN_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
