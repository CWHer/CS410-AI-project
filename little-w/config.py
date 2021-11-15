from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "xxx": 1,
    "final_reward": 10,
}


NETWORK_CONFIG = {
    "periods_num": 5,
    "num_channels": 256,
    "num_res": 4,
    "board_size": 10 * 20,
    "action_size": 64,
}

MCTS_CONFIG = {

}

DATA_GEN_CONFIG = {

}

TRAIN_CONFIG = {
    "l2_weight": 1e-4,
    "learning_rate": 0.001,
    "check_point": "check_point"

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
