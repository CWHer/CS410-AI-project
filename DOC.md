## D3QN说明文档

### Features

每个智能体独立使用DQN训练

训练智能体 i 时，将其他智能体直接看作环境的一部分

- [x] Residual Network
- [ ] Set proper rewards
- [x] Multiprocessing
- [ ] TensorBoard real-time monitoring
- [x] Single player mode
- [x] Two player mode
- [ ] comments
- [ ] ...



### 代码架构

大致如下图所示（不完全一致）

![](assets/code-structure.png)



### 文件说明

| Env文件夹    | snakes_3v3环境               |
| ------------ | ---------------------------- |
| snake_env/   | 官方提供的env                |
| simulator.py | 官方env的包裹类              |
| utils.py     | 自定义features/action/reward |

| Agent文件夹      |          |
| ---------------- | -------- |
| network.py       | D3QN     |
| network_utils.py | 网络杂项 |

| Train_utils文件夹 |                |
| ----------------- | -------------- |
| game.py           | 自我博弈，对战 |
| replay_buffer.py  | 存储数据       |

| File            | Description          |
| --------------- | -------------------- |
| train.py (main) | 训练pipeline         |
| utils.py        | 可视化及其它辅助函数 |
| config.py       | 超参数设置           |



### 训练Pipeline

1. Self Play生成数据，保存在replay buffer
2. 数据量足够后开始训练model
3. 几轮训练后与best net对局，胜率>55%则更新模型



### 特征选取

`[(6 + 2k), 10, 20]`

- 我方蛇全身位置（当前阶段以及前k-1个阶段）
- 敌方蛇全身位置（当前阶段以及前k-1个阶段）
- 当前控制蛇头位置
- 当前控制蛇全身位置
- 我方当前其余蛇头位置
- 敌方当前蛇头位置
- 所有豆的位置
- [0, 1]矩阵表示当前时间（$t=\text{step}/200$）

NOTE: 无法保证还原出每一节的方向

```python
# k不能太小，k=2无法还原成蛇头方向
x   9  10  11		x   1   2   3
1   8   7   6		7   6   5   4	
2   3   4   5		8   9  10  11
```



### 奖励设计

参考：后浪-snakes3v3

- 设计单条蛇的零和奖励：
  $$
  \hat r_i=r_i-\bar r'
  $$
  上式中$r_i$为第 i 条蛇与上一帧的长度差，$\bar r'$为敌方3条蛇长度差的平均 

- 为鼓励团队配合，奖励设计为
  $$
  r_i^\star=(1-\alpha)\cdot \hat r_i + \alpha\cdot\bar {\hat r}
  $$
  上式中$\alpha$为超参团队奖励系数，$\bar {\hat r}$为团队零和奖励的平均

- 加入赢局奖励 final_reward=20

  赢得 比赛获得奖励，输则给予负奖励

- [ ] reward随时间变化

NOTE: 使用总长度差作为每轮的奖励时，死亡会带来持续多轮的负收益



