"""
    This part of code is the Q learning brain, which is a brain of the agent.
    All decisions are made in here.

    View more on my tutorial page: https://morvanzhou.github.io/tutorials/
    代码复用，QLearning和Sarsa继承同一个RL类
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

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
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s_last, action, reward, s_temp):
        self.check_state_exist(s_temp)
        q_last = self.q_table.loc[s_last, action]
        if s_temp != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[s_temp, :].max()
        else:
            q_target = reward
        self.q_table.loc[s_last, action] = q_last + self.lr * (q_target - q_last)


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s_last, action_last, reward, s_temp, action_temp):
        self.check_state_exist(s_temp)
        q_last = self.q_table.loc[s_last, action_last]
        if s_temp != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[s_temp, action_temp]
        else:
            q_target = reward
        self.q_table.loc[s_last, action_last] = q_last + self.lr * (q_target - q_last)
