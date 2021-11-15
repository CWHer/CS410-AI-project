from collections import namedtuple

import numpy as np
from agent.mcts import MCTSPlayer
from agent.network_utils import ObsEncoder
from config import MDP_CONFIG
from env.chooseenv import make
from icecream import ic


def selfPlay(net):
    """[summary]
    self play and gather experiences

    Args:
        net ([type]): [description]

    Returns:
        states [List[np.ndarray]]: [description]
        mcts_probs [List[np.ndarray]]: [description]
        values [List[float]]: [description]
    """
    # TODO: initialize seeds (torch, np, random)
    env = make("snakes_3v3")
    state = env.reset()
    encoder = ObsEncoder()

    data_buffer, episode_len = [], 0
    total_reward = np.zeros(6)
    players = [MCTSPlayer(net, i) for i in range(2)]

    while True:
        episode_len += 1
        encoder.add(state)

        # NOTE: getAction MUST NOT change env & encoder
        # NOTE: data = (features(states), mcts_probs)
        actions, data = zip(
            *[player.getAction(env, encoder)
              for player in players])
        data_buffer.extend(data)
        joint_actions = actions[0] + actions[1]
        for i in range(2):
            players[i].updateRoot(joint_actions)

        joint_actions = env.encode(joint_actions)
        next_state, reward, done, _, info = env.step(joint_actions)
        total_reward += np.array(reward)
        state = next_state

        if done:
            score0 = sum(total_reward[:3])
            score1 = sum(total_reward[3:])
            ic(score0, score1)
            # NOTE: drop data of draws
            if score0 == score1:
                return [], [], []
            winner = int(score1 > score0)
            ic(winner)
            # TODO: modify rewards
            states, mcts_probs = zip(*data_buffer)
            values = [
                MDP_CONFIG.final_reward * ((i & 1) ^ winner ^ 1)
                for i in range(episode_len * 2)]
            ic(len(values), len(states))
            return states, mcts_probs, values


def contest(net0, net1):
    pass
