from agent.utils import Actions, ActionsFilter
from icecream import ic
from env.chooseenv import make


ic(Actions.inv(Actions.UP))
ic(ActionsFilter.Act2Idx([Actions.RIGHT, Actions.RIGHT, Actions.RIGHT]))
ic(ActionsFilter.Idx2Arr(63))

last_actions = ActionsFilter.extractActions(
    [[[0, 1], [0, 2]], [[0, 1], [1, 1]], [[0, 1], [8, 1]]])
ic(last_actions)
indices = ActionsFilter.genActions(last_actions)
for index in indices:
    ic(ActionsFilter.Idx2Act(index))
