import copy
import itertools
from collections import deque
from enum import Enum, unique

import numpy as np

from config import MDP_CONFIG, NETWORK_CONFIG


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
        in_channels = NETWORK_CONFIG.in_channels
        board_height = MDP_CONFIG.board_height
        board_width = MDP_CONFIG.board_width
        features = np.zeros(
            (in_channels, board_height, board_width))

        # positions of my body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in team_indices:
                features[i][tuple(zip(*self.states[i][j]))] = 1

        # positions of opponent's body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in opponent_indices:
                features[periods_num + i][tuple(zip(*self.states[i][j]))] = 1

        # current position of controlled snake head
        x, y = self.states[-1][idx][0]
        features[periods_num * 2][x][y] = 1

        # current position of controlled snake body
        features[periods_num * 2 + 1][tuple(zip(*self.states[-1][idx]))] = 1

        # current positions of team rest head
        for i in itertools.filterfalse(
                lambda x: x == idx, team_indices):
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 2][x][y] = 1

        # current positions of opponent's head
        for i in opponent_indices:
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 3][x][y] = 1

        # positions of all the beans
        features[-2][tuple(zip(*self.states[-1][bean_index]))] = 1

        # progress
        features[-1, ...] = num_step / MDP_CONFIG.total_step

        # extract
        x, y = self.states[-1][idx][0]
        width = MDP_CONFIG.input_width
        obs_features = np.zeros(
            (in_channels, width, width))

        def fn(x, mod): return (x + mod - width // 2) % mod
        for k in range(in_channels):
            obs_features[k] = np.array([
                [features[k][fn(x + i, board_height)]
                    [fn(y + j, board_width)] for j in range(width)]
                for i in range(width)])

        # rotate
        action = ActionsFilter.extractActions([self.states[-1][idx]])
        rot_num = ActionsFilter.rot_nums[action[0]]
        obs_features = np.array(
            [np.rot90(s, rot_num) for s in obs_features])

        return features, obs_features


@unique
class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def rot90(action, k):
        """[summary]
        similar to np.rot90
        NOTE: counter-clockwise
        """
        ROT_MAP = [
            [0, 1, 2, 3], [2, 3, 1, 0],
            [1, 0, 3, 2], [3, 2, 0, 1]]
        return Actions(ROT_MAP[k % 4][action.value])


class ActionsFilter():
    """[summary]
    4 ^ 3 = 64 actions in total for each player.
    However, (4 - 1) ^ 3 = 27 actions are legal,
    as head CAN NOT move towards its body

    """
    # NOTE: delta = head_pos - next_pos
    actions_map = {
        (-1, 0): Actions.UP,
        (1, 0): Actions.DOWN,
        (0, -1): Actions.LEFT,
        (0, 1): Actions.RIGHT
    }

    # rotate current action to UP (rot90)
    rot_nums = {
        Actions.UP: 0,
        Actions.RIGHT: 1,
        Actions.DOWN: 2,
        Actions.LEFT: 3,
    }

    @staticmethod
    def genActions(last_action):
        """[summary]
        e.g. last_actions = [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]
        generate legal actions according to last_actions
        """
        actions = [Actions.LEFT, Actions.UP, Actions.RIGHT]
        inv_rot_num = 4 - ActionsFilter.rot_nums[last_action]
        valid_actions = [
            Actions.rot90(action, inv_rot_num)
            for action in actions
        ]
        indices = [action.value for action in valid_actions]
        return indices

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
