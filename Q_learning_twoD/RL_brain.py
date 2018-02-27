"""
    Q learning brain —— a brain of the agent.
"""
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # 打乱位置，对相同值情况的随机化选择
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 对操作得到的新状态的反馈计算和table表的数值更新
    def learn(self, temp_state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[temp_state, action]
        if next_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        # 修改这个action在之前状态下的值，相当于给出反馈
        self.q_table.loc[temp_state, action] += self.lr * (q_target - q_predict)  # update

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
