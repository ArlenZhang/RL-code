"""
    搭建流程：
        get_date() : 更新记忆库数据，供训练使用
        create_logits() : 直接将模型搭建和数据(get_data准备的数据存储在类变量域)结合在一起，得到logits输出
        create_loss() : 同神经网络使用
        create_optimize() : ...
    作者: Longyin Zhang
    日期: 2018.3.2

"""
import numpy as np
import tensorflow as tf
import util
np.random.seed(1)
tf.set_random_seed(1)

"""
    Deep Q Network 类
"""
class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False):
        self.q_eval = self.q_next = self.cost = self.memory_counter = self._train_op = self.loss = self.q_target = self.s_ = self.s =\
            self.sess = self.replace_target_op = None
        # ========== 初始化模型的超参数
        self.lr = learning_rate  # 学习效率
        self.batch_size = batch_size  # 批处理量
        self.c_names_e = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.c_names_t = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.n_l1 = 10  # 第一层神经元个数
        self.w_initializer = tf.random_normal_initializer(0., 0.3)
        self.b_initializer = tf.constant_initializer(0.1)

        # ========== 配置QLearning计算过程中用到的参数
        self.gamma = reward_decay   # 反馈衰减值
        self.epsilon_max = e_greedy  # 决定选择使用最优action还是随机action的比重
        self.epsilon_increment = e_greedy_increment  # 随模型训练时间增加而越发偏向最优action选择，增大epsilon
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.n_actions = n_actions  # 可选动作
        self.n_features = n_features  # 观察值转成n_features列的数据
        self.memory_size = memory_size  # 记忆库的大小
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 创建初始记忆库

        # ========== 初始化全局变量
        self.learn_step_counter = 0  # learn一次就累加1
        self.replace_target_iter = replace_target_iter  # 学习隔replace_target_iter步更新一次target值
        self.cost_steps = []  # 存储各个批次网络的误差值

        # ========== 初始化模型
        self.add_placeholder()
        self._build_net()
        self.create_loss_opt()  # 创建计算loss和optimizer的操作
        self.create_target_params_opt()  # 创建更新target网络参数的操作

        # ========== tensorflow 会话
        self.session_summary(output_graph=output_graph)

    def add_placeholder(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

    """
        创建两个结构相同参数不同的神经网络 target_n 和 eval_n
    """
    def _build_net(self):
        # 创建评估网络
        # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            # 第一层网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_l1], initializer=self.w_initializer,
                                     collections=self.c_names_e)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=self.b_initializer, collections=self.c_names_e)
                if self.s is None:
                    input("s")
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二层神经网络
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_actions], initializer=self.w_initializer,
                                     collections=self.c_names_e)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=self.b_initializer,
                                     collections=self.c_names_e)
                self.q_eval = tf.matmul(l1, w2) + b2

        # 创建target目标神经网络
        with tf.variable_scope('target_net'):
            # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
            # 第一层网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_l1], initializer=self.w_initializer,
                                     collections=self.c_names_t)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=self.b_initializer, collections=self.c_names_t)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # 第二层神经网络
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_actions], initializer=self.w_initializer,
                                     collections=self.c_names_t)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=self.b_initializer,
                                     collections=self.c_names_t)
                self.q_next = tf.matmul(l1, w2) + b2

    # 计算损失
    def create_loss_opt(self):
        # 定义损失函数，也就是QLearning中的(Q_target - Q_predict)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # optimizer 反向传播过程，学习神经网络参数
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    # 用Eval网络参数更新Target网络参数
    def create_target_params_opt(self):
        # 定义获取target和eval网络的参数 w 和 b的操作
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    # tensorflow 会话
    def session_summary(self, output_graph):
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("../../graphs/DQN/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    """
        QLearning 算法应用部分
    """
    def learn(self):
        # 学习每经过300次,更换一次target网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget参数被更新\n')

        # 从记忆库中选择batch_size的数据学习，这是RL应用神经网络的关键
        # arr中每一行的存储: self.memory[index, :] = np.hstack((s_last, [action, reward], s_temp))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 从batch_memory中提取四组信息 state, action, reward, state_
        states_, states, actions, rewards = util.get_data(batch_memory, self.n_features)

        # 对当前随机参数的模型输入state根据原有的模型参数计算得到两组q值，分别为 Q评估 和 Q现实
        # 返回的是状态对应的分值，相当于查Q表的功能，用两个网络作为一个Q表能力更好
        q_eval, q_next = self.sess.run(
            [self.q_eval, self.q_next],
            feed_dict={
                self.s_: states_,  # 每个训练记录的state_值
                self.s: states,  # 每个训练记录的state 值
            })

        # Sarsa算法计算Q_target
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)

        # 将states, q_target送入神经网络, states再次经神经网络计算的到q_eval 再与q_target计算loss值并训练网络
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: states, self.q_target: q_target})
        self.cost_steps.append(self.cost)

        # 提高e-greedy的epsilon值，提高对模型的信任
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 更新记忆库
    def store_transition(self, s_last, action, reward, s_temp):
        if not hasattr(self, 'memory_counter') or self.memory_counter is None:
            self.memory_counter = 0
        transition = np.hstack((s_last, [action, reward], s_temp))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    """
            选择根据观察值 state，将state数据作为eval神经网络的输入数据， 
        输出不同action的，取最大value对应的action.
    """
    def choose_action(self, observation):
        # 将1维观察值转成2维,神经网络设置的nfeatures就是2
        observation = observation[np.newaxis, :]
        # 90% 最优
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_steps)), self.cost_steps)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
