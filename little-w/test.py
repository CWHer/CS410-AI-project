from env.chooseenv import make
import numpy as np
from icecream import ic
import copy


def get_join_actions():
    actions = np.random.randint(4, size=6)
    return actions


env = make("snakes_3v3", conf=None)
state = env.reset()

while True:
    joint_action = get_join_actions()
    action = env.encode(joint_action)
    # NOTE: env supports deepcopy :)
    env_copy = copy.deepcopy(env)
    # NOTE: return values
    #   (all_observes, reward, done, info_before, info_after)
    next_state, reward, done, _, info = env.step(action)
    ic(next_state, info)
    next_state_copy, reward, done, _, info_copy = env_copy.step(action)
    ic(next_state_copy, info_copy)
    break
