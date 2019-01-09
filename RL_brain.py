"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):# 和 QLearningTable 中的代码一样
    #初始化
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate#學習率
        self.gamma = reward_decay#衰減率
        self.epsilon = e_greedy#貪婪度

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)#初始化Qtable
    
    # 检测 state 是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)# 检测本 state 是否在 q_table 中存在(见后面标题内容)
        # 選擇action
        if np.random.rand() < self.epsilon:# 选择 Q value 最高的 action
            state_action = self.q_table.loc[observation, :]
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机选择 action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args): # 每种的都有点不同, 所以用 pass
        pass


# off-policy離綫
class QLearningTable(RL): # 继承了父类 RL
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)#繼承

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy在綫
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)#繼承

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':# 下个 state 不是 终止符
            q_target = r + self.gamma * self.q_table.loc[s_, a_]# q_target 基于选好的 a_ 而不是 Q(s_) 的最大值
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)   # 更新 q_table
