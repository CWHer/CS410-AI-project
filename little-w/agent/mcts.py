import copy

import numpy as np
from config import MCTS_CONFIG, MDP_CONFIG
from icecream import ic
from tqdm import tqdm

from utils import plotHeatMaps, plotLines, printError, timeLog

from .mcts_utils import TreeNode
from .network_utils import ObsEncoder
from .utils import Actions, ActionsFilter, canEat


class MCTS():
    def __init__(self, net, turn) -> None:
        self.net = net
        # NOTE: [1, 0] each round if self.turn == 1
        self.turn = turn  # who am I
        self.root = TreeNode(None, None, 1.0)

    def updateRoot(self, joint_actions):
        """[summary]
        update root with joint_actions

        """
        self.root = TreeNode(None, None, 1.0)

        # FIXME: TODO: bug with stochastic env
        # FIXME: root.parent = None
        # self.root = self.root.transfer(
        #     joint_actions[:3]).transfer(joint_actions[3:])
        # if self.root is None:
        #     self.root = TreeNode(None, None, 1.0)

    @staticmethod
    def plotActionProb(actions_probs):
        probs = np.zeros((len(Actions), ) * 3)
        for action, prob in actions_probs:
            probs[tuple(ActionsFilter.Idx2Arr(action))] = prob
        plotHeatMaps(ic(probs), Actions)

    def __search(self, env,
                 encoder, score_board):
        """[summary]
        perform one search, which contains
        1. selection     2. expansion
        3. simulation    4. backpropagate
        """
        # >>>>> selection
        node, turn = self.root, None
        # is_stochastic: whether eat beans
        done, is_stochastic = False, False
        trajectory_rewards = []
        while not done \
                and turn is None:
            actions = [None] * 2
            for i in [self.turn, self.turn ^ 1]:
                if node.isLeaf:
                    turn = i
                    break
                node = node.select()
                actions[i] = ActionsFilter.Idx2Arr(node.action)

            if all(actions):
                joint_actions = env.encode(actions[0] + actions[1])
                # TODO: How to set reward?
                # HACK: BUG: FIXME: reborn with random position
                #   It is actually not random, BUT CAN I USE THIS FEATURE?
                #   Or should we just DO NOT make those decisions?
                # NOTE: For now, I just leverage on determinism

                if node.isLeaf and \
                        canEat(env, joint_actions):
                    # NOTE: the positions of beats are random too
                    #   What about sample several states and mix (predicted) pdfs
                    prev_env = copy.deepcopy(env)
                    prev_encoder = copy.deepcopy(encoder)
                    is_stochastic = True
                    # ic(actions)

                next_state, reward, done, *_ = env.step(joint_actions)
                encoder.add(next_state)
                score_board.add(reward)
                trajectory_rewards.append(score_board.reward())

        # >>>>> expansion & simulation
        if not done:
            # NOTE: use network to predict new node
            #   which yields prior_prob for each leaf node
            #    and v in [-value_scale, value_scale] for current node

            if is_stochastic:
                # HACK: FIXME: what if a snake eat a beat then die, is reward 0 ?

                # NOTE: FIXME: if next_state is stochastic,
                #   what about sample many states and average predicted pdfs.
                states = [encoder.encode(turn)]
                for _ in range(MDP_CONFIG.state_sample_num - 1):
                    env_copy = copy.deepcopy(prev_env)
                    encoder_copy = copy.deepcopy(prev_encoder)
                    next_state, *_ = env_copy.step(joint_actions)
                    # env_copy.draw_board()
                    encoder_copy.add(next_state)
                    states.append(encoder_copy.encode(turn))

                policy, value = self.net.predict(np.stack(states))
                policy, value = policy.mean(axis=0), value.mean()
            else:
                # determinism is awesome!
                policy, value = self.net.predict(encoder.encode(turn))

            # NOTE: what you did last step is not necessarily the true directions,
            #   as snakes may die and then reborn with new directions
            value = value.item()
            last_actions = encoder.getLastActions()
            last_actions = last_actions[:3] if turn == 0 else last_actions[3:]
            indices = ActionsFilter.genActions(last_actions)
            # ic(list(map(ActionsFilter.Idx2Act, indices)))
            actions_probs = [
                (action, policy[action]) for action in indices]
            node.expand(actions_probs)
            # self.plotActionProb(actions_probs)
        else:
            turn = self.turn ^ 1
            # game is over
            winner = score_board.getWinner()
            value = (winner != -1) * MDP_CONFIG.final_reward
            # backpropagate opponent first
            if winner == self.turn:
                value = -value

        # >>>>> backpropagate
        # MC estimate
        while not node is None:
            if turn == self.turn:
                value *= MCTS_CONFIG.gamma
                if trajectory_rewards:
                    # backpropagate opponent first
                    value += trajectory_rewards[-1] * \
                        (1 if self.turn == 1 else -1)
                    trajectory_rewards.pop()
            node.update(value)
            value = -value
            turn ^= 1
            node = node.parent

        # debug
        # self.root.printDebugInfo()

    def search(self, env,
               encoder, score_board):
        # NOTE: make sure root node is correct
        # for _ in tqdm(range(MCTS_CONFIG.search_num)):
        for _ in range(MCTS_CONFIG.search_num):
            self.__search(
                copy.deepcopy(env),
                copy.deepcopy(encoder),
                copy.deepcopy(score_board))

        # debug
        self.root.printDebugInfo()

        # choose actions
        actions, vis_cnt = list(zip(
            *[(child.action, child.getVisCount())
              for child in self.root.children]))
        # NOTE: TEMPERATURE controls the level of exploration
        #   N ^ (1 / T) = exp(1 / T * log(N))
        probs = MCTS_CONFIG.inv_temperature * \
            np.log(np.array(vis_cnt) + 1e-10)
        probs = np.exp(probs - np.max(probs))
        probs /= np.sum(probs)

        # debug
        # self.plotActionProb(zip(actions, probs))

        return actions, probs


class MCTSPlayer():
    """[summary]
    NOTE: snakes3v3 env is stochastic 
    1. snake may reborn with random position (though it is deterministic actually)
        It is really terrible as (4 - 1) ^ 3 actions would CHANGE!
        HACK: FIXME: For now, I leverage on the determinism for training.
            Since rebirth is deterministic, it is sufficient to store only one action set each MCTS node.
        NOTE: FIXME: Here I assume the agent would not take advantage of this feature.
            That is, agent trained under this env has generalization ability.
        HACK: An alternative is to just stop here and return reward
    2. the position of beans are random
        HACK: FIXME: I try to fix this by introducing more samples 
            For prior probability, what about sample many states and average predict pdfs.
            For search in MCTS, just follow old MCTS strategy as this only affects states not actions.
    """

    def __init__(self, net, turn) -> None:
        self.turn = turn
        self.mcts = MCTS(net, turn)

    def updateRoot(self, joint_actions):
        self.mcts.updateRoot(joint_actions)

    @timeLog
    def getAction(self, env,
                  encoder, score_board,
                  is_train=False):
        """[summary]

        Returns:
            (action, mcts_probs)
        """
        actions, probs = self.mcts.search(env, encoder, score_board)

        if is_train:
            # NOTE: add noise for exploration
            noise = np.random.dirichlet(
                np.full(len(probs), MCTS_CONFIG.dirichlet_alpha))
            action = np.random.choice(
                actions,
                p=((1 - MCTS_CONFIG.dirichlet_eps) * probs
                   + MCTS_CONFIG.dirichlet_eps * noise))

            # debug
            # plotLines([(probs, "original"),
            #            (((1 - MCTS_CONFIG.dirichlet_eps) * probs
            #              + MCTS_CONFIG.dirichlet_eps * noise), "adding noise")])
        else:
            action = np.random.choice(actions, p=probs)
            # action = actions[np.argmax(probs)]

        mcts_probs = np.zeros(MDP_CONFIG.action_size)
        mcts_probs[np.array(actions)] = probs

        return action, (encoder.encode(self.turn), mcts_probs)
