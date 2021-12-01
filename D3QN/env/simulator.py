from .snake_env.chooseenv import make
from .utils import ActionsFilter, ObsEncoder, ScoreBoard


class Simulator():
    """[summary]
    # TODO: implement more APIs

    a wrapper of snakes3v3
    NOTE: support customized features, reward and actions (utils.py)
    """

    def __init__(self) -> None:
        self.env = make("snakes_3v3", conf=None)
        self.encoder = None
        self.score_board = None
        self.num_step = 0

    def drawBoard(self):
        self.env.draw_board()

    def reset(self):
        state = self.env.reset()

        self.num_step = 0
        self.encoder = ObsEncoder()
        self.encoder.add(state)
        self.score_board = ScoreBoard()

        features = [self.encoder.encode(
            idx=i + 1, num_step=self.num_step) for i in range(6)]
        return features

    def validActions(self):
        """[summary]
        return valid actions' indices under current state
        """
        # NOTE: what you did last step is not necessarily the true directions,
        #   as snakes may die and then reborn with new directions
        last_actions = self.encoder.getLastActions()
        indices = map(ActionsFilter.genActions, last_actions)
        return list(indices)

    def step(self, joint_action):
        """[summary]
        NOTE: joint_action = [0, 4] ^ 6
        """
        self.num_step += 1

        joint_action = self.env.encode(joint_action)
        # (all_observes, reward, done, info_before, info_after)
        next_state, reward, done, *info = self.env.step(joint_action)

        self.encoder.add(next_state)
        self.score_board.add(reward)

        features = [self.encoder.encode(
            idx=i + 1, num_step=self.num_step) for i in range(6)]
        # NOTE: below is reward of player0
        reward = [self.score_board.getReward(i, done) for i in range(6)]

        return features, reward, done, info
