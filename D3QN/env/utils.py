import copy
from collections import deque
from enum import Enum, unique
from itertools import filterfalse, product

import numpy as np
from config import MDP_CONFIG, NETWORK_CONFIG
from icecream import ic

from utils import printError


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

    def encode(self, turn, num_step) -> np.ndarray:
        """[summary]
        features selection
        """
        bean_index = 0
        my_indices = [1, 2, 3] if turn == 0 else [4, 5, 6]
        opponent_indices = [4, 5, 6] if turn == 0 else [1, 2, 3]

        periods_num = NETWORK_CONFIG.periods_num
        features = np.zeros(
            (NETWORK_CONFIG.in_channels,
             MDP_CONFIG.board_height,
             MDP_CONFIG.board_width))

        # positions of my body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in my_indices:
                features[i][tuple(zip(*self.states[i][j]))] = 1
        # current positions of my head
        for i, idx in enumerate(my_indices):
            x, y = self.states[-1][idx][0]
            features[periods_num + i][x][y] = 1

        # positions of opponent's body in previous k periods
        #   (including current period)
        for i in range(periods_num):
            for j in opponent_indices:
                features[periods_num + i +
                         3][tuple(zip(*self.states[i][j]))] = 1
        # current positions of opponent's head
        for i in opponent_indices:
            x, y = self.states[-1][i][0]
            features[periods_num * 2 + 3][x][y] = 1

        # positions of all the beans
        features[-3][tuple(zip(*self.states[-1][bean_index]))] = 1

        # all 0 if turn == 0 else all 1
        if turn == 1:
            features[-2, ...] = np.ones_like(features[-2, ...])

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

    @staticmethod
    def fliplr(action):
        if action.value < 2:
            return action
        else:
            return Actions(action.value ^ 1)


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

    @staticmethod
    def rot90(index, k):
        actions = ActionsFilter.Idx2Act(index)
        return ActionsFilter.Act2Idx(
            list(map(lambda x: Actions.rot90(x, k), actions)))

    @staticmethod
    def fliplr(index):
        actions = ActionsFilter.Idx2Act(index)
        return ActionsFilter.Act2Idx(
            list(map(Actions.fliplr, actions)))

    @staticmethod
    def Act2Idx(actions):
        actions = list(map(lambda x: x.value, actions))
        return actions[2] + actions[1] * 4 + actions[0] * 16

    @staticmethod
    def Idx2Arr(index):
        return [index // 16, index // 4 % 4, index % 4]

    @staticmethod
    def Idx2Act(index):
        return list(map(Actions, ActionsFilter.Idx2Arr(index)))

    @staticmethod
    def genActions(last_actions):
        """[summary]
        e.g. last_actions = [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]
        generate legal actions according to last_actions
        """
        ic.configureOutput(includeContext=True)
        printError(
            len(last_actions) != 3,
            ic.format("too many actions in genActions!"))
        ic.configureOutput(includeContext=False)

        inv_actions = list(map(Actions.inv, last_actions))
        # HACK: FIXME: BUG: the commented code isn't working
        # actions = product(*[filterfalse(
        #     lambda x: x == inv_actions[i], Actions)
        #     for i in range(3)])
        actions = product(*(list(filterfalse(
            lambda x: x == inv_actions[i], Actions))
            for i in range(3)))
        indices = [ActionsFilter.Act2Idx(action) for action in actions]
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


class ScoreBoard():
    def __init__(self) -> None:
        self.scores = np.zeros(6)
        self.score0, self.score1 = 0, 0

    def add(self, rewards):
        self.scores += np.array(rewards)
        self.score0 = sum(self.scores[:3])
        self.score1 = sum(self.scores[3:])

    def getWinner(self):
        return -1 if self.score0 == self.score1 \
            else int(self.score1 > self.score0)

    def getReward(self, done):
        """[summary]

        Returns:
            reward [type]: [description]. instant reward of player0
        """
        # TODO: add final reward

        return MDP_CONFIG.c_reward * \
            (self.score0 - self.score1)
