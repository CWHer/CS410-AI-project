import copy
from collections import deque
from enum import Enum, unique
from itertools import filterfalse, product
import itertools

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
    def genActions(last_action):
        """[summary]
        e.g. last_actions = [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]
        generate legal actions according to last_actions
        """
        inv_action = Actions.inv(last_action)
        actions = list(filterfalse(
            lambda x: x == inv_action, Actions))
        indices = [action.value for action in actions]
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
        self.last_rewards = [0] * 6
        self.score0, self.score1 = 0, 0

    def add(self, rewards):
        self.scores += np.array(rewards)
        self.last_rewards = rewards
        self.score0 = sum(self.scores[:3])
        self.score1 = sum(self.scores[3:])

    def showResults(self):
        print(self.scores)

    def getWinner(self):
        return -1 if self.score0 == self.score1 \
            else int(self.score1 > self.score0)

    def getReward(self, idx, done):
        """[summary]

        Returns:
            reward [type]: [description]. instant reward of player0
        """
        if done:
            winner = self.getWinner()
            return 0 if winner == -1 \
                else MDP_CONFIG.final_reward * \
                (1 if winner == 0 and idx < 3 else -1)

        opponent_r = sum(
            self.last_rewards[3:] if idx < 3 else self.last_rewards[:3]) / 3
        team_r = sum(self.last_rewards) / 3 - opponent_r
        r = self.last_rewards[idx] - opponent_r
        return (1 - MDP_CONFIG.c_reward) * r + MDP_CONFIG.c_reward * team_r
