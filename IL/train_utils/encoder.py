import copy
import itertools
from collections import deque

import numpy as np
from config import NETWORK_CONFIG, MDP_CONFIG


class ObsEncoder():
    def __init__(self) -> None:
        self.states = deque(
            maxlen=NETWORK_CONFIG.periods_num)

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
