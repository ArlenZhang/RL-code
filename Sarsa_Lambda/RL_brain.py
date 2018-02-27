"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # 创建Q表
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

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
        self.check_state_exist(observation)
        # 选择action
        if np.random.rand() < self.epsilon:  # 90% use best action
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # 将每个状态下的action对应的索引下标打乱顺序
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            # 依旧根据最大值选取，同QLearning
            action = state_action.idxmax()
        else:
            # 选择随机action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.7):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 对还不存在的state添加到Q表
            new_row = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(new_row)

            # 更新标记表
            self.eligibility_trace = self.eligibility_trace.append(new_row)

    def learn(self, s_last, action_last, reward, s_temp, action_temp):
        self.check_state_exist(s_temp)
        q_last = self.q_table.loc[s_last, action_last]
        if s_temp != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_temp, action_temp]
        else:
            q_target = reward
        # Sarsa(LAMBDA)不之处在于下面的过程
        # 对于state-action对的访问次数进行标记
        # 方案 1:
        # self.eligibility_trace.loc[s_last, action_last] += 1  # 弊端在于累加容易过大
        # 方案 2:
        self.eligibility_trace.loc[s_last, :] *= 0
        self.eligibility_trace.loc[s_last, action_last] = 1  # 保持该state-action的访问当前state的为峰值

        # Q update, 每一次对全表中标记过的全体Q表数据进行更新
        self.q_table += self.lr * (q_target - q_last) * self.eligibility_trace

        # 更新之后对标记信息进行衰减，保证了距离当前action越近的标记的价值越大
        self.eligibility_trace *= self.gamma*self.lambda_
