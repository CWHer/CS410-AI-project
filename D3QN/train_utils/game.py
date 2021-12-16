import random

import numpy as np
import torch
from env.simulator import Simulator
from icecream import ic
from utils import printError


def epsilonGreedy(q_values, indices, epsilon=0.1):
    ic.configureOutput(includeContext=True)
    printError(
        q_values.shape[-1] != len(indices),
        ic.format("shapes do not match!"))
    ic.configureOutput(includeContext=False)

    if np.random.rand() < epsilon:
        return random.choice(indices)
    else:
        return indices[q_values.argmax(axis=1).item()]


def selfPlay(net, epsilon, seed):
    """[summary]
    self play and gather experiences

    Args:
        net ([type]): [description]

    Returns:
        states [List[np.ndarray]]: [description]
        mcts_probs [List[np.ndarray]]: [description]
        values [List[float]]: [description]
    """
    # initialize seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_buffer = []
    env = Simulator()
    state = env.reset()

    while True:
        # NOTE: last_actions is used in enhanceData
        prev_actions = env.encoder.getLastActions()
        indices = env.validActions()
        actions = [None] * 6
        for i in range(6):
            q_values = net.predict(state[i])
            actions[i] = epsilonGreedy(q_values, indices[i], epsilon)

        next_state, reward, done, _ = env.step(actions)

        # collect data
        for i in range(6):
            data_buffer.append(
                (state[i], reward[i], actions[i],
                 prev_actions[i], next_state[i], done))

        state = next_state
        # debug
        # env.drawBoard()

        if done:
            env.score_board.showResults()
            return list(zip(*data_buffer))


def contest(net0, net1, seed):
    """[summary]
    contest between net0 and net1

    Returns:
        int: [description]. winner
    """
    # initialize seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = Simulator()
    state = env.reset()
    players = [net0, net1]

    while True:
        indices = env.validActions()
        actions = [None] * 6
        for i in range(6):
            q_values = players[int(i > 2)].predict(state[i])
            actions[i] = epsilonGreedy(
                q_values, indices[i], epsilon=0)

        next_state, reward, done, _ = env.step(actions)
        state = next_state
        # debug
        # env.drawBoard()

        if done:
            env.score_board.showResults()
            return env.score_board.getWinner()