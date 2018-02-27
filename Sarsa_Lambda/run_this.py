"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from RL_code.Sarsa_Lambda.maze_env import Maze
from RL_code.Sarsa_Lambda.RL_brain import SarsaLambdaTable

def update():
    for episode in range(100):
        # 初始化环境，获取环境状态
        s_last = env.reset()
        # RL choose action based on observation
        action_last = RL.choose_action(str(s_last))
        # initial all zero eligibility trace, eligibility表具有和Q表一样的内容，并做0初始化
        RL.eligibility_trace *= 0
        while True:
            # 刷新图形界面环境
            env.render()
            s_temp, reward, done = env.step(action_last)
            action_temp = RL.choose_action(str(s_temp))
            # RL learn
            RL.learn(str(s_last), action_last, reward, str(s_temp), action_temp)
            s_last = s_temp
            action_last = action_temp
            # break while loop when end of this episode
            if done:
                break
    # 结束游戏
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
