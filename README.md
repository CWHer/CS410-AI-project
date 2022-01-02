# CS410-AI-project

本分支用于本地evaluate:thinking:

目前包括: random, rl(DDPG), greedy, heuristic, dqn(mlp), rot_dqn, IL, defense_dqn, defense_IL

训练好的模型可以在[Link](https://drive.google.com/drive/folders/1vOMKE5JC1PCZ6HCpZIJ9L2SpiNcOY5yG?usp=sharing)找到，其中包含DQN模型（用于rot_dqn和defense_dqn）和IL模型（用于IL和defense_IL），将model文件移入对应的`agent/xxx`文件夹，并修改`network.py`中加载模型的代码即可运行。

注：`dqn_models`包含若干效果较好的DQN模型。`il`仅包含一个效果较好的IL模型以及其拆分后的两个part（测试平台有文件大小限制:weary:）。

## 如何运行

```python
python evaluation_local.py --my_ai rl --opponent random
python run_log.py --my_ai "random" --opponent "rl"
```



## Description of branches

| Branch Name | Content                    |
| ----------- | -------------------------- |
| greedy      | greedy snake               |
| AlphaZero   | Alpha Zero snake           |
| D3QN        | (deserted)                 |
| single-D3QN | DQN without rotation trick |
| rot-D3QN    | DQN with rotation trick    |
| IL          | Imitaion Learning          |

Note: defense module is in `main`.

