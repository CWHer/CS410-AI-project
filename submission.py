
"""[summary]
Greedy Agent by Yuri
"""

action_choices = {
    (1, 0, 0, 0): [-1, 0],
    (0, 1, 0, 0): [1, 0],
    (0, 0, 1, 0): [0, -1],
    (0, 0, 0, 1): [0, 1]}


def distance(observation, position):
    for snake_id in range(2, 8):
        if position in observation[snake_id]:
            return 999
    min_dis = 9999
    for i in observation[1]:
        x = abs(i[0] - position[0])
        y = abs(i[1] - position[1])
        x_size = observation["board_height"]
        y_size = observation["board_width"]
        dis = min(x, x_size - x) + min(y, y_size - y)
        # dis=x+y
        min_dis = min(dis, min_dis)
    return min_dis


def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    snake_id = observation["controlled_snake_index"]
    # last_direction=observation["last_direction"]
    x_size = observation["board_height"]
    y_size = observation["board_width"]
    # delta_x=(observation[snake_id][0][0]-observation[snake_id][1][0]+x_size)%x_size
    # delta_y=(observation[snake_id][0][1]-observation[snake_id][1][1]+y_size)%y_size
    # if delta_x==x_size-1:last_direction=-2#up
    # elif delta_x==1:last_direction=2#down
    # elif delta_y==y_size-1:last_direction=-1#left
    # else:last_direction=1#right
    head = observation[snake_id][0]
    max_value = 9999
    for action in action_choices:
        new_position = head.copy()
        new_position[0] = (
            new_position[0] + action_choices[action][0] + x_size) % x_size
        new_position[1] = (
            new_position[1] + action_choices[action][1] + y_size) % y_size
        value = distance(observation, new_position)
        if value <= max_value:
            best_action = action
            max_value = value
    return [list(best_action)]
