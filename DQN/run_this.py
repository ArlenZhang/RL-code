from RL_code.DQN.maze_env import Maze
from RL_code.DQN.RL_brain import DeepQNetwork

# maze 环境
def run_maze():
    step = 0
    # 指定训练300次观察cost再决定训练批次
    for episode in range(300):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()
            env.after(100)
            # RL choose action based on observation
            # 不再通过Q表选取动作，而是把当前观察值作为神经网络参数计算得到一组Q值，取最优
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # 记忆库在不断更新，最近的知识更重要，替换了以往知识
            RL.store_transition(observation, action, reward, observation_)
            # 200步骤以前都是在建立记忆库(只是存储数据环境+操作), 200 以后每5步学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将当前观察值更新
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
    # 在tensorboard中观察结果
    # tensorboard --logdir="/home/arlenzhang/Desktop/RL/RL_Pro/logs"
