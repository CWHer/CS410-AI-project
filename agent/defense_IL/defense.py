import random
import numpy as np


class Info:
    def __init__(self, idx=-1, step=-1) -> None:
        self.idx = idx
        self.step = step

    def empty(self):
        return self.idx == -1

    def nextStep(self):
        return Info(self.idx, self.step + 1)


class DefensiveBoard:
    actions_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, observation) -> None:
        self.snakes = dict()
        self.beans = observation[1]
        self.width = observation["board_width"]
        self.height = observation["board_height"]
        self.control_snake = observation["controlled_snake_index"] - 2

        self.board = \
            [[Info() for _ in range(self.width)]
             for _ in range(self.height)]

        for idx in range(6):
            self.snakes[idx] = observation[idx + 2].copy()
            for x, y in self.snakes[idx]:
                self.board[x][y] = Info(idx, step=0)

        self.construct()

    def construct(self):
        """[summary]
        BFS distance field

        NOTE:
            node[0] -> Info
            node[1] -> list[int, int]: position
        """

        global_step = 0
        # NOTE: shorter snake first, as it is more aggressive
        q = [(Info(i, step=0), self.snakes[i][0])
             for i in self.snakes.keys()]
        q.sort(key=lambda x: len(self.snakes[x[0].idx]))

        while q:
            node = q.pop(0)

            # step forward
            if node[0].step == global_step:
                global_step += 1
                for snake in self.snakes.values():
                    if len(snake) >= global_step:
                        x, y = snake[-global_step]
                        self.board[x][y] = Info()

            for next_x, next_y in self.genNextPos(*node[-1]):
                info = self.board[next_x][next_y]
                # occupy the empty space
                if info.empty():
                    q.append((node[0].nextStep(), [next_x, next_y]))
                    self.board[next_x][next_y] = node[0].nextStep()

    def genNextPos(self, x, y):
        ret = [
            [(x + dx + self.height) % self.height,
             (y + dy + self.width) % self.width]
            for dx, dy in self.actions_map]
        return ret

    def getAction(self, head, target, step) -> int:
        """[summary]
        backtracking
        """
        index = self.control_snake
        now_position = target
        while step > 1:
            step -= 1
            legal_position = []
            for x, y in self.genNextPos(*now_position):
                info = self.board[x][y]
                if info.idx == index \
                        and info.step == step:
                    legal_position.append([x, y])
            now_position = random.choice(legal_position)
        return self.extractAction(head, now_position)

    def extractAction(self, s, t) -> int:
        delta = np.array(t) - np.array(s)
        if (np.abs(delta) > 1).any():
            delta = -np.clip(delta, -1, 1)
        return self.actions_map.index(tuple(delta))


def getDefenseAction(observation):
    board = DefensiveBoard(observation)
    snake_index = board.control_snake
    # seek safe defense action
    for interval in range(len(board.snakes[snake_index]) // 2):
        for body in range(1, len(board.snakes[snake_index]) - interval + 1):
            target = board.snakes[snake_index][-body]
            info = board.board[target[0]][target[1]]
            if info.idx == snake_index \
                    and info.step == body + interval:
                return board.getAction(
                    board.snakes[snake_index][0], target, info.step)
    return None
