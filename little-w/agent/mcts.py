import copy
import itertools
from collections import namedtuple

import numpy as np
from numpy.core.numeric import indices
from config import MCTS_CONFIG
from tqdm import tqdm

from utils import timeLog

from .mcts_utils import TreeNode
from .network_utils import ObsEncoder
from .utils import ActionsFilter


class MCTS():
    def __init__(self, net, turn) -> None:
        self.net = net
        self.turn = turn  # who am I
        self.root = TreeNode(None, None, 1.0)

    def plotDebugInfo(self):
        # TODO: display using heatmap
        pass

    def __search(self, env, encoder: ObsEncoder):
        """[summary]
        perform one search, which contains
        1. selection     2. expansion
        3. simulation    4. backpropagate

        """
        # >>>>> selection
        node, turn = self.root, self.turn
        for step in itertools.count():
            actions = [None, None]
            for i in range(2):
                if node.isLeaf:
                    break

                actions[i], node = node.select()
                turn ^= 1

            if all(actions):
                joint_actions = env.encode(actions[0] + actions[1])
                # TODO: How to deal with reward?
                # HACK: BUG: FIXME: reborn with random position
                #   It is actually not random, BUT CAN I USE THIS FEATURE?
                #   Or should we just DO NOT make those decisions?
                # NOTE: For now, I just leverage on determinism
                # NOTE: the positions of beats are random too
                if node.isLeaf and canEat():
                    # NOTE: sample several states and mix (predicted) pdfs
                    env_copy = copy.deepcopy(env)
                next_state, reward, done, _, info = env.step(joint_actions)
                encoder.add(next_state)

            if done:
                break

        # >>>>> expansion & simulation
        # NOTE: what you did last step is not necessarily the true directions,
        #   as snakes may die and then reborn with new directions
        # TODO: extract features from state
        if not done:
            if (np.array(reward) > 0).any():
                # NOTE: FIXME: if next_state is stochastic,
                #   what about sample many states and average predict pdfs.
                # TODO: sample and mix pdfs
                pass
            else:
                # deterministic is awesome!
                policy, value = self.net.predict(encoder.encode(turn))

            last_actions = encoder.getLastActions()
            indices = ActionsFilter.genActions(last_actions)
            policy = policy[np.array(indices)]
            node.expand(zip(indices, policy))
        else:
            # game is over
            value = xxxx

        # NOTE: DO NOT forget to negate reward of opponent
        if turn != self.turn:
            value = -value

        # >>>>> backpropagate
        while not node is None:
            node.update(value)
            value = -value
            if turn == 0:
                value *= MCTS_CONFIG.gamma
            node = node.parent

        # TODO: debug

    def search(self, env, encoder):
        # NOTE: make sure root node is correct
        for _ in tqdm(range(MCTS_CONFIG.search_num)):
            self.__search(
                copy.deepcopy(env),
                copy.deepcopy(encoder))

        # TODO: debug
        # self.root.printDebugInfo(

        # TODO: choose actions

    def updateRoot(self, joint_actions):
        """[summary]
        update root with joint_actions

        """
        # FIXME: TODO: bug with stochastic env
        self.root = self.root.transfer(
            joint_actions[:3]).transfer(joint_actions[3:])
        if self.root is None:
            self.root = TreeNode(None, None, 1.0)


class MCTSPlayer():
    """[summary]
    NOTE: snakes3v3 env is stochastic 
    1. snake may reborn with random position (though it is deterministic actually)
        HACK: FIXME: For now, I just leverage on the determinism for training.
            Since rebirth is deterministic, it is sufficient to store only one action set each MCTS node.
        NOTE: FIXME: Here I assume the agent would not take advantage of this feature.
            That is, agent trained under this env has generalization.
    2. the position of beans are random
        HACK: FIXME: I try to fix this by introducing more samples 
            For prior probability, what about sample many states and average predict pdfs.
            For search in MCTS, just follow old strategy as it only affects states not actions.
    """

    def __init__(self, net, turn) -> None:
        # TODO

        self.mcts = MCTS(net, turn)

    @timeLog
    def getAction(self, env, encoder, is_train=False):
        """[summary]

        Returns:
            (action, prob)
        """
        # debug
        # Data = namedtuple("data", "state mcts_prob")
        # features = encoder.encode(turn=self.turn)
        # mcts_prob, _ = self.net.predict(features)
        # data = Data(features, mcts_prob)
        # return [0, 0, 0], data

        actions, probs = self.mcts.search(env)

        if is_train:
            # TODO: add noise

            pass
        else:
            # choose best action
            action = actions[np.argmax(probs)]

        # debug
        # TODO: plot heat map

        # TODO:
        Data = namedtuple("data", "state mcts_prob turn")
        return action, Data(xxx)

    def updateRoot(self, joint_actions):
        self.mcts.updateRoot(joint_actions)
