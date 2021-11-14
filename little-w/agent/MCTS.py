from numpy.random.mtrand import gamma
from .MCTS_utils import TreeNode
from config import MCTS_CONFIG
import tqdm
import copy
import itertools
from utils import timeLog
import numpy as np


class MCTS():
    def __init__(self, turn) -> None:
        self.turn = turn  # who to move
        self.root = TreeNode(None, None, 1.0)

    def plotDebugInfo(self):
        # TODO: display using heatmap
        pass

    def __search(self, env):
        """[summary]
        perform one search, which contains
        1. selection     2. expansion
        3. simulation    4. backpropagate

        """
        # >>>>> selection
        node, turn = self.root, self.turn
        for step in itertools.count():
            actions = []
            for i in range(2):
                actions[i], node = node.select()
                turn ^= 1
                if node.isLeaf:
                    break

            joint_actions = env.encode(actions[0] + actions[1])
            # TODO
            next_state, reward, done, _, info = env.step(joint_actions)
            if done:
                break

        # >>>>> expansion & simulation
        # TODO
        # HACK: DO NOT forget to negate reward of opponent
        if not done:
            actions_probs, value = self.net.predict()
            node.expand
        else:
            # game is over
            value = xxxx

        # >>>>> backpropagate
        while not node is None:
            node.update(value)
            value = -value
            if turn == 0:
                value *= gamma
            node = node.parent

        # TODO: debug

    def search(self, env):
        # NOTE: make sure root node is correct
        for _ in tqdm.tqdm(range(xxxx)):
            self.__search(copy.deepcopy(env))

        # TODO: debug
        # self.root.printDebugInfo(

        # TODO: choose actions

    def updateRoot(self, joint_actions):
        """[summary]
        update root with joint_actions

        """
        self.root = self.root.transfer(
            joint_actions[:3]).transfer(joint_actions[3:])
        if self.root is None:
            self.root = TreeNode(None, None, 1.0)


class MCTSPlayer():
    def __init__(self) -> None:
        self.mcts = MCTS(xxx)

    @timeLog
    def getAction(self, env, is_train=False):
        """[summary]

        Returns:
            (action, prob)
        """

        actions, probs = self.mcts.search(env)

        if is_train:
            # TODO: add noise

            pass
        else:
            # choose best action
            action = actions[np.argmax(probs)]

        # debug
        # TODO: plot heat map

        return action, probs[action]

    def updateRoot(self, joint_actions):
        self.mcts.updateRoot(joint_actions)
