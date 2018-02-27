"""
Qlearning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from RL_code.Q_learning_twoD.maze_env import Maze
from RL_code.Q_learning_twoD.RL_brain import QLearningTable

N_EPISODE = 100

def update():
    for episode in range(N_EPISODE):
        # 初始化对环境的观察
        observation = env.reset()
        while True:
            # 刷新环境
            env.render()
            # 根据观察到的状态选择动作
            action = RL.choose_action(str(observation))
            # 更新环境,图形界面反馈用户
            observation_, reward, done = env.step(action)
            # print("\r\n原state和action和reward和新state: ")
            # print(str(observation))
            # print(action)
            # print(reward)
            # print(observation_)
            # input()
            # 从上一步的转移过程得到的新的状态计算反馈信息并学习
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
