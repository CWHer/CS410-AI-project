from enum import Enum, unique
from itertools import filterfalse, product

import numpy as np
from icecream import ic


@unique
class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def inv(action):
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
    def act2idx(actions):
        actions = list(map(lambda x: x.value, actions))
        return actions[2] + actions[1] * 4 + actions[0] * 16

    @staticmethod
    def idx2act(index):
        actions = [index // 16, index // 4 % 4, index % 4]
        return list(map(Actions, actions))

    @staticmethod
    def genActions(last_actions):
        """[summary]
        e.g. last_actions = [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]
        generate legal actions according to last_actions

        """
        inv_actions = list(map(Actions.inv, last_actions))
        actions = product(*[filterfalse(
            lambda x: x == inv_actions[i], Actions)
            for i in range(3)])
        indices = [ActionsFilter.act2idx(action) for action in actions]
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


def beEaten():
    # whether snakes will be eaten after taking current action
    # NOTE: I am not sure if this is necessary
    pass


def canEat():
    # whether snakes can eat bean after taking current action
    # TODO
    pass


# ic(Actions.inv(Actions.UP))
# ic(ActionsFilter.act2idx([Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]))
# ic(ActionsFilter.idx2act(63))
# last_actions = ActionsFilter.extractActions(
#     [[[0, 1], [0, 2]], [[0, 1], [1, 1]], [[0, 1], [8, 1]]])
# ic(last_actions)
# ic(ActionsFilter.genActions(last_actions))
