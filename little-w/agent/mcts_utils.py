import math

from config import MCTS_CONFIG
from icecream import ic

from utils import printError


class PUCT():
    """[summary]
    PUCT functor
    """

    def __init__(self, prior_prob) -> None:
        self.prob = prior_prob
        self.Q, self.N = 0, 0

    def update(self, v):
        # self.Q = (self.Q * self.N + v) / (self.N + 1)
        self.N += 1
        self.Q += (v - self.Q) / self.N

    def U(self, c_puct, total_N):
        return c_puct * self.prob * \
            math.sqrt(total_N) / (1 + self.N)

    def PUCT(self, total_N):
        # TODO: add puct decay
        return self.Q() + self.U(MCTS_CONFIG.c_puct, total_N)


class TreeNode():
    """[summary]
    MCTS tree node

    NOTE: a node is either unvisited or fully expanded
    NOTE: (4 - 1) ^ 3 = 27 actions (children)
    """

    def __init__(self, parent, action, prior_prob) -> None:
        """[summary]
        s (parent) --> a (action) --> s' (self)
        NOTE: here action is an index
        """
        self.parent = parent
        self.action = action
        self.puct = PUCT(prior_prob)
        self.children = None

    def getVisCount(self):
        return self.puct.N

    def PUCT(self, total_N):
        return self.puct.PUCT(total_N)

    def update(self, v):
        self.puct.update(v)

    @property
    def isLeaf(self):
        return self.children is None

    @property
    def isRoot(self):
        return self.parent is None

    def select(self):
        """[summary]
        select the child node with maximal PUCT
        """
        total_N = self.getVisCount()
        return max(self.children,
                   key=lambda x: x.PUCT(total_N))

    def expand(self, actions_probs) -> None:
        """[summary]
        (fully) expand this node with prior probabilities

        Args:
            actions_probs (list): [description]. a list of (action, prior_prob)
        """
        self.children = [
            TreeNode(self, action, prior_prob)
            for action, prior_prob in actions_probs]

    def transfer(self, action):
        """[summary]
        transfer to next state
        """
        if self.children is None:
            return None
        for child in self.children:
            if child.action == action:
                return child
        printError(True, "fail to find child!")

    def printDebugInfo(self):
        total_N = self.getVisCount()
        for child in self.children:
            print("Action: {} with Q {:>6f}, PUCT {:>6f}, N {}".format(
                child.action, child.puct.Q, child.PUCT(total_N), child.getVisCount()))
