import numpy as np
import pandas as pd
import time
# pandas 也是处理数据用的一个包
np.random.seed(2)  # 产生伪随机数
N_STATES = 6  # 有多少个状态，-----o 也就是从开始到宝藏之间有多少状态
ACTIONS = ["left", "right"]  # 有几种操作
EPSILON = 0.9  # 百分之10进行随机 百分之90选择最优
ALPHA = 0.1  # 学习率
LAMBDA = 0.9  # 衰减程度
MAX_EPISODES = 26  # 训练次数
FRESH_TIME = 0.3  # 0.3秒走一步

# 创建Q表格
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
        )
    print(table)
    return table

# 创建选择动作的函数
def choose_action(state, table):
    state_actions = table.iloc[state, :]
    # 大于0.9的占据10% 用随机
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        # state_actions : id  left_val  right_val 选择较大的值的动作
        action_name = state_actions.idxmax()
    return action_name

# 创建环境对当前state对应的操作action的反馈
def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES - 2:
            state = 'terminal'
            reward = 1
        else:
            state = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state = state
        else:
            state = state - 1
    return state, reward

def update_env(state, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# 强化学习主程序
def r_learning():
    table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        temp_state = 0
        is_terminated = False
        update_env(temp_state, episode, step_counter)
        while not is_terminated:
            action = choose_action(temp_state, table)
            state, reward = get_env_feedback(temp_state, action)
            # print("\r\n原state和action和reward和新state: ")
            # print(temp_state)
            # print(action)
            # print(reward)
            # print(state)
            # input()
            q_predict = table.ix[temp_state, action]
            if state != 'terminal':
                q_target = reward + LAMBDA * table.iloc[state, :].max()
            else:
                q_target = reward
                is_terminated = True

            table.ix[temp_state, action] += ALPHA * (q_target - q_predict)
            temp_state = state
            # 更新
            update_env(temp_state, episode, step_counter+1)
            step_counter += 1
    return table

if __name__ == "__main__":
    q_table = r_learning()
    print('\r\nQ-table: \n')
    print(q_table)
