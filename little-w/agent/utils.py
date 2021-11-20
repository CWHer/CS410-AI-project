import copy
from enum import Enum, unique
from itertools import filterfalse, product

import numpy as np
from config import MDP_CONFIG
from icecream import ic

from utils import printError


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


# NOTE: delta = head_pos - next_pos
actions_map = {
    (-1, 0): Actions.UP,
    (1, 0): Actions.DOWN,
    (0, -1): Actions.LEFT,
    (0, 1): Actions.RIGHT
}


class ActionsFilter():
    """[summary]
    4 ^ 3 = 64 actions in total for each player.
    However, (4 - 1) ^ 3 = 27 actions are legal,
    as head CAN NOT move towards its body

    """
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
            actions.append(actions_map[tuple(delta)])
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

    def reward(self):
        """[summary]

        Returns:
            reward [type]: [description]. instant reward of player0
        """
        return MDP_CONFIG.c_reward * \
            (self.score0 - self.score1)


def beEaten():
    # whether snakes will be eaten after taking current action
    # NOTE: I am not sure if this is necessary
    pass


def canEat(env, joint_actions):
    # whether snakes can eat bean after taking current action
    # TODO: profile this naive method
    _, reward, *_ = copy.deepcopy(env).step(joint_actions)
    return (np.array(reward) > 0).any()
