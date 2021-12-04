import numpy as np

head_action = {"up": [2, 1, 3], "right": [1, 3, 0], "down": [3, 0, 2], "left": [0, 2, 1]}


def transform(feature, direction):
    fea = feature
    if direction == "up":
        fea = feature
    elif direction == "right":
        # 逆时针旋转90度
        fea = np.transpose(feature)
        fea = fea[::-1]
    elif direction == "down":
        # 逆时针旋转180度
        fea = np.transpose(feature)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
    elif direction == "left":
        # 逆时针旋转270度
        fea = np.transpose(feature)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
    return fea


def head_and_obs(state, idx):
    snake = state[idx]
    head = snake[0]
    tail = snake[1]
    height, width = state["board_height"], state["board_width"]
    if ((head[0] == ((tail[0] + 1) % height)) and (head[1] == tail[1])):  # up
        direction = "up"
    elif ((head[0] == ((tail[0] + height - 1) % height)) and (head[1] == tail[1])):  # down
        direction = "down"
    elif (head[0] == tail[0]) and (head[1] == ((tail[1] + 1) % width)):  # right
        direction = "right"
    elif (head[0] == tail[0]) and (head[1] == ((tail[1] + width - 1) % width)):  # left
        direction = "left"

    obs = state
    feature = []
    agent = idx
    fea_screen = np.zeros((height, width))  # 特征0：食物是1，身体是-1
    perfect_feature = np.zeros((height, height))  # 特征0：食物是1，身体是-1

    snake_head = []
    snake_length = 0
    for num in obs.keys():
        if num == 1:
            data = obs[num]
            for coordinate in data:
                fea_screen[coordinate[0], coordinate[1]] = 1
        elif num == agent:  # 自己的信息
            data = obs[num]
            snake_head = data[0]
            snake_length = [len(data)]
            for coordinate in data:
                fea_screen[coordinate[0], coordinate[1]] = -1
        elif isinstance(num, int):
            data = obs[num]
            head = 0
            for coordinate in data:
                if head == 0:
                    fea_screen[coordinate[0], coordinate[1]] = -2
                    head = 1
                else:
                    fea_screen[coordinate[0], coordinate[1]] = -1
    for x in range(height):
        for y in range(height):
            dx = (snake_head[0] + height + (x - 5)) % height
            dy = (snake_head[1] + width + (y - 5)) % width
            perfect_feature[x, y] = fea_screen[dx, dy]
    perfect_feature = transform(perfect_feature, direction)
    perfect_feature = perfect_feature.reshape(-1).tolist()
    # perfect_feature = perfect_feature[:24] + perfect_feature[25:]
    # feature.append(perfect_feature + snake_length)
    feature.append(perfect_feature)
    return direction, feature


def again_action(state, action):
    # action：0:向下走;1:向上走;2:向左走;3:向右走
    # 所有非尾巴
    danger_position = []
    width = state["board_width"]
    height = state["board_height"]
    my_control = state["controlled_snake_index"]
    my_control_head = state[my_control][0]
    for num in state.keys():
        if num == 1:
            pass
        elif isinstance(num, int):
            snake = state[num]
            for body in snake[:-1]:
                danger_position += [body]
    head_0 = [(my_control_head[0] + height - 1) % height, my_control_head[1]]  # 向下
    head_1 = [(my_control_head[0] + height + 1) % height, my_control_head[1]]  # 向上
    head_2 = [my_control_head[0], (my_control_head[1] + width - 1) % width]  # 向左
    head_3 = [my_control_head[0], (my_control_head[1] + width + 1) % width]  # 向右
    pre_next_head = {0: head_0, 1: head_1, 2: head_2, 3: head_3}
    next_head = pre_next_head[action]
    if next_head in danger_position:  # 下一个位置是危险位置，重新选择动作
        for next_action, next_position in pre_next_head.items():
            if next_position in danger_position:
                pass
            else:
                action = next_action
                break

    return action


def Judge_done(state, next_state):
    my_control = state["controlled_snake_index"]
    DONE = 0
    snake_length = len(state[my_control])
    next_snake_length = len(next_state[my_control])
    if next_snake_length < snake_length:
        DONE = 1
        # print(state, next_state)
    return DONE

