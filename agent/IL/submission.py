import os
import sys
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

# fmt: off
from features import MDP_CONFIG, ObsEncoder
from network import Imitator
# fmt: on

model = Imitator()
file_path = os.path.dirname(
    os.path.abspath(__file__)) + "/model_3_40.pth"
# file_path = os.path.dirname(
#     os.path.abspath(__file__))
model.load(file_path)

encoder = ObsEncoder()
num_step = -1
chosen_actions = [None for _ in range(6)]
actions_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

sys.path.pop(-1)  # just for safety


def chooseAction(state, probs):
    # NOTE: remove actions that would hit any body (not including tails)
    # from icecream import ic
    # ic.disable()
    # ic(q_values)
    height, width = 10, 20
    danger_position = np.zeros((height, width))
    agent = state["controlled_snake_index"]
    team_id = [2, 3, 4] if agent <= 4 else [5, 6, 7]
    opponent_id = [2, 3, 4] if agent > 4 else [5, 6, 7]

    # body
    for i in range(2, 8):
        danger_position[tuple(zip(*state[i][:-1]))] = 1

    # opponent attack
    for i in opponent_id:
        if len(state[i]) < len(state[agent]):
            x, y = state[i][0]
            danger_position[x][y] = 1
            for dx, dy in actions_map:
                danger_position[
                    (x + dx + height) % height][(y + dy + width) % width] = 1

    # collide with teammate
    for i in team_id:
        if i >= agent:
            break
        x, y = state[i][0]
        dx, dy = chosen_actions[i - 2]
        danger_position[
            (x + dx + height) % height][(y + dy + width) % width] = 1

    my_head = state[agent][0]
    for i in range(4):
        x, y = actions_map[i]
        x = (my_head[0] + height + x) % height
        y = (my_head[-1] + width + y) % width
        probs[0][i] *= (not danger_position[x][y])

    # ic(q_values)
    return probs.argmax(axis=1).item()


def my_controller(observation, *_):
    # 2, 3, 4, 5, 6, 7
    agent = observation["controlled_snake_index"]

    global num_step, encoder
    # BUG: HACK: cause bug when self play
    if agent == 2 or agent == 5:
        num_step += 1
        if num_step == MDP_CONFIG.total_step:
            # restart
            num_step = 0
            encoder = ObsEncoder()
        encoder.add([observation])

    state = encoder.encode(agent - 1, num_step)
    probs = model.predict(state)
    action = chooseAction(observation, probs)
    chosen_actions[agent - 2] = actions_map[action]
    action_mat = [[0, 0, 0, 0]]
    action_mat[0][action] = 1
    return action_mat
