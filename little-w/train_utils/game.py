import random

import numpy as np
import torch
from agent.mcts import MCTSPlayer
from agent.network_utils import ObsEncoder
from agent.utils import ScoreBoard
from config import MDP_CONFIG
from env.chooseenv import make
from icecream import ic


def selfPlay(net, seed):
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

    env = make("snakes_3v3")
    state = env.reset()
    encoder = ObsEncoder()
    data_buffer, episode_len = [], 0
    score_board = ScoreBoard()
    players = [MCTSPlayer(net, i) for i in range(2)]

    while True:
        episode_len += 1
        encoder.add(state)
        # NOTE: getAction MUST NOT change env, encoder & score_board
        # NOTE: data = (features(states), mcts_probs)
        actions, data = zip(
            *[player.getAction(env, encoder,
                               score_board, is_train=True)
              for player in players])
        data_buffer.extend(data)
        joint_actions = actions[0] + actions[1]
        for i in range(2):
            players[i].updateRoot(joint_actions)

        joint_actions = env.encode(joint_actions)
        # NOTE: reward might be negative
        next_state, reward, done, *_ = env.step(joint_actions)
        score_board.add(reward)
        state = next_state
        # debug
        # env.draw_board()

        if done:
            winner = score_board.getWinner()
            # NOTE: drop data of draws
            if winner == -1:
                return [], [], []
            # ic(winner, score_board.scores)
            # TODO: modify rewards
            states, mcts_probs = zip(*data_buffer)
            # TODO: FIXME: this is biased
            values = [
                MDP_CONFIG.final_reward *
                (1 if (i & 1) == winner else -1)
                for i in range(episode_len * 2)]
            # ic(len(values), len(states))
            return states, mcts_probs, values


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

    env = make("snakes_3v3")
    state = env.reset()
    encoder = ObsEncoder()
    score_board = ScoreBoard()
    players = [MCTSPlayer(net0, 0), MCTSPlayer(net1, 1)]

    while True:
        encoder.add(state)
        # NOTE: getAction MUST NOT change env, encoder & score_board
        actions, _ = zip(
            *[player.getAction(env, encoder,
                               score_board, is_train=True)
              for player in players])
        joint_actions = actions[0] + actions[1]
        for i in range(2):
            players[i].updateRoot(joint_actions)

        joint_actions = env.encode(joint_actions)
        # NOTE: reward might be negative
        next_state, reward, done, *_ = env.step(joint_actions)
        score_board.add(reward)
        state = next_state
        # debug
        # env.draw_board()

        if done:
            # ic(score_board.scores)
            return score_board.getWinner()
