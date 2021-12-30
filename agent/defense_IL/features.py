import copy
import itertools
from collections import deque, namedtuple
from enum import Enum, unique

import numpy as np

MDP_CONFIG = {
    "board_height": 10,
    "board_width": 20,
    "board_size": 10 * 20,
    "action_size": 4,
    "total_step": 200,
}

NETWORK_CONFIG = {
    "periods_num": 3,
    "in_channels":  None,
    "num_channels": 256,
    "num_res": 5,
}
NETWORK_CONFIG["in_channels"] = \
    NETWORK_CONFIG["periods_num"] * 2 + 6


MDP_CONFIG_TYPE = namedtuple("MDP_CONFIG", MDP_CONFIG.keys())
MDP_CONFIG = MDP_CONFIG_TYPE._make(MDP_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())


class ObsEncoder():
    def __init__(self) -> None:
        self.states = deque(
            maxlen=NETWORK_CONFIG.periods_num)

    def getLastActions(self):
        """[summary]
        get actions of last state
        """
        return ActionsFilter.extractActions(self.states[-1][1:])

    def add(self, raw_state):
        """[summary]
        add (fill) new state to deque 
        NOTE: new state is repeated to FILL deque

        Args:
            raw_state ([type]): [description] output of env
        """
        raw_state = copy.deepcopy(raw_state[0])
        state = [raw_state[i] for i in range(1, 8)]
        if self.states:
            self.states.popleft()
        while (len(self.states) <
               NETWORK_CONFIG.periods_num):
            self.states.append(state)

    def encode(self, idx, num_step) -> np.ndarray:
        """[summary]
        features selection
        """
        bean_index = 0
        team_indices = [1, 2, 3] if idx <= 3 else [4, 5, 6]
        opponent_indices = [4, 5, 6] if idx <= 3 else [1, 2, 3]

        periods_num = NETWORK_CONFIG.periods_num
        features = np.zeros(
            (NETWORK_CONFIG.in_channels,
             MDP_CONFIG.board_height,
             MDP_CONFIG.board_width))
        for i in range(periods_num):
            for j in team_indices:
                features[i][tuple(zip(*self.states[i][j]))] = 1
        for i in range(periods_num):
            for j in opponent_indices:
                features[periods_num + i][tuple(zip(*self.states[i][j]))] = 1
        x, y = self.states[-1][idx][0]
        features[periods_num * 2][x][y] = 1
        features[periods_num * 2 + 1][tuple(zip(*self.states[-1][idx]))] = 1
        for i in itertools.filterfalse(
                lambda x: x == idx, team_indices):
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 2][x][y] = 1
        for i in opponent_indices:
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 3][x][y] = 1
        features[-2][tuple(zip(*self.states[-1][bean_index]))] = 1
        features[-1, ...] = num_step / MDP_CONFIG.total_step

        return features


@unique
class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def inv(action):
        return Actions(action.value ^ 1)


class ActionsFilter():
    # NOTE: delta = head_pos - next_pos
    actions_map = {
        (-1, 0): Actions.UP,
        (1, 0): Actions.DOWN,
        (0, -1): Actions.LEFT,
        (0, 1): Actions.RIGHT
    }

    @staticmethod
    def extractActions(snakes):
        """[summary]
        extract last actions from state

        Args:
            snakes ([type]): [description]. contains lists of snake
        """
        actions = []
        for snake in snakes:
            delta = np.array(snake[0]) - np.array(snake[1])
            if (np.abs(delta) > 1).any():
                delta = -np.clip(delta, -1, 1)
            actions.append(ActionsFilter.actions_map[tuple(delta)])
        return actions
