import argparse
import os
import random

import numpy as np
import torch
from icecream import ic
from tabulate import tabulate
from torch.distributions import Categorical

from env.chooseenv import make

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_actions(state, algo, indexs):
    if algo == "random":
        # random agent
        actions = np.random.randint(4, size=3)
        return actions
    elif algo == "rl":
        # rl agent
        from agent.rl.submission import agent, get_observations
        obs = get_observations(
            state[0], indexs, obs_dim=26, height=10, width=20)
        logits = agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array(
            [Categorical(out).sample().item()
             for out in logits])
        return actions
    elif algo == "greedy":
        from agent.greedy.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [])[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "heuristic":
        from agent.heuristic.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "mlp_dqn":
        from agent.mlp_dqn.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "rot_dqn":
        from agent.rot_dqn.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "IL":
        from agent.IL.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "defense_dqn":
        from agent.defense_dqn.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions
    elif algo == "defense_IL":
        from agent.defense_IL.submission import my_controller
        actions = [0] * 3
        for i in range(len(actions)):
            action = my_controller(state[indexs[i]], [], None)[0]
            actions[i] = action.index(1)
        return actions

    ic.configureOutput(prefix="BUG -> ")
    ic.configureOutput(includeContext=True)
    print(f"NO {algo} algorithm!")
    raise NotImplementedError()


def get_join_actions(obs, algo_list):
    indexs = [0, 1, 2, 3, 4, 5]
    first_action = get_actions(obs, algo_list[0], indexs[:3])
    second_action = get_actions(obs, algo_list[1], indexs[3:])
    actions = np.zeros(6)
    actions[:3] = first_action[:]
    actions[3:] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):

    total_reward = np.zeros(6)
    num_win = np.zeros(3)

    for i in range(1, episode + 1):
        episode_reward = np.zeros(6)

        state = env.reset()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)

            next_state, reward, done, _, info = env.step(
                env.encode(joint_action))
            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    num_win[0] += 1
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print("total_reward: ", total_reward)
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(np.sum(total_reward[:3]), 2), np.round(np.sum(total_reward[3:]), 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    env_type = 'snakes_3v3'

    game = make(env_type, conf=None)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rl", help="rl/random")
    parser.add_argument("--opponent", default="random", help="rl/random")
    parser.add_argument("--episode", default="100")
    args = parser.parse_args()

    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list,
             episode=int(args.episode), verbose=False)
