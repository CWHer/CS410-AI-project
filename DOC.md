## D3QN说明文档

### Features

作为单智能体，统一决策

NOTE: 目前不考虑不合法动作

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

![](assets/alpha_go_zero_cheat_sheet.png)



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

`[(9 + 2k), 10, 20]`

- 我方蛇全身位置（当前阶段以及前k-1个阶段）
- 敌方蛇全身位置（当前阶段以及前k-1个阶段）
- 我方当前蛇头位置（按顺序分为3个channel，区分每条:snake:各自的动作）
- 我方当前蛇全身位置（按顺序分为3个channel，区分每条:snake:各自的动作）
- 敌方当前蛇头位置
- 所有豆的位置
- [0, 1]矩阵表示当前时间（$t=\text{step}/200$）

NOTE: 无法保证还原出每一节的方向

目前 k=5

```python
# k不能太小，k=2无法还原成蛇头方向
x   9  10  11		x   1   2   3
1   8   7   6		7   6   5   4	
2   3   4   5		8   9  10  11
```



### 奖励设计

- $r_i$为第 i 条蛇与上一帧的长度差

  $r=\bar r-\bar r'$，$\bar r $为我方平均，$\bar r'$为敌方平均

- 加入赢局奖励 final_reward=20

- [ ] reward随时间变化

NOTE: 使用总长度差作为每轮的奖励时，死亡会带来持续多轮的负收益

