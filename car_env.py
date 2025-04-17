#自定义环境（含障碍物 & 加速度）
import matplotlib
matplotlib.use("TkAgg")  # 添加这一行切换后端

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class CarEnv(gym.Env):
    def __init__(self):   #初始化函数
        super(CarEnv, self).__init__()   #调用父类（gym.Env）的初始化函数，确保环境正确继承了父类的一些必要功能。
        self.max_speed = 1.0  #小车的最大速度为 1.0（单位可以是 m/s 或其他，这里是简化单位）。
        self.dt = 1.0    #时间步长为 1.0，也就是说每执行一步，模拟时间增加 1 个单位。
        self.acceleration = 0.1   #加速度为 0.1，每次加速操作都会以这个速度改变速度向量。
        self.velocity = np.array([0.0, 0.0]) #初始化小车的速度为 [0.0, 0.0]，即初始时静止。
        self.position = np.array([0.0, 0.0])  #小车的初始位置设为坐标原点 [0.0, 0.0]。
        self.target = np.array([9.0, 9.0])  #目标位置为 [9.0, 9.0]，小车的任务就是移动到这个位置。
        self.obstacles = [((4, 4), (6, 6))]  # 定义了一个障碍物，它是一个矩形，从 (4,4) 到 (6,6)，小车不能穿过这个区域。
        self.path = []  # 这个列表用于记录小车的行驶轨迹，可以在可视化中用来画出移动路径。

        # 动作空间：加速度方向 [上，下，左，右，停止] 0: 向上加速 1: 向下加速2: 向左加速3: 向右加速4: 停止（不加速）
        self.action_space = spaces.Discrete(5)  #动作空间是离散的，共 5 个动作
        # 状态空间：位置x,y 和速度x,y
        self.observation_space = spaces.Box(low=np.array([0, 0, -1, -1]),
                                            high=np.array([10, 10, 1, 1]),
                                            dtype=np.float32)    #spaces.Box 表示这个状态空间是一个连续的多维空间。
        #状态空间是一个连续空间，表示 [位置x, 位置y, 速度x, 速度y]。范围定义为：位置在 [0, 10] 之间速度在 [-1, 1] 之间


    def reset(self):   #这是环境的重置函数。当开始一个新的 episode（回合）时，Gym 会调用这个函数来初始化环境状态。
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.path = [self.position.copy()] #重新初始化路径记录，记录下初始位置。这里用 .copy() 是为了避免之后位置更新影响 path 中已保存的位置
        return np.concatenate([self.position, self.velocity])
        #返回当前的观测状态：把位置和速度拼接成一个数组 [x, y, vx, vy]。
    def step(self, action):
        # 更新速度
        if action == 0:   # 加速向上
            self.velocity[1] += self.acceleration
        elif action == 1: # 加速向下
            self.velocity[1] -= self.acceleration
        elif action == 2: # 加速向左
            self.velocity[0] -= self.acceleration
        elif action == 3: # 加速向右
            self.velocity[0] += self.acceleration
        elif action == 4: # 停止
            self.velocity = np.array([0.0, 0.0])

        # 限速  #确保速度不超过设定的最大值，防止小车“加速飞了”。
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)

        # 位置更新 用速度乘以时间步长来更新位置，并限制小车不出边界 [0, 10]。同时记录轨迹，
        self.position += self.velocity * self.dt
        self.position = np.clip(self.position, 0, 10)
        self.path.append(self.position.copy())

        # 检查碰撞
        done = False
        reward = -np.linalg.norm(self.position - self.target)
        #初始奖励为负的欧氏距离，表示越接近目标奖励越高（但还是负数）。这是一种“稀疏+引导”的奖励形式，适合 DQN 收敛。

        for (x_min, y_min), (x_max, y_max) in self.obstacles:
            if x_min <= self.position[0] <= x_max and y_min <= self.position[1] <= y_max:
                reward -= 50
                done = True  # 撞上障碍结束

        def is_at_target(position, target, threshold=0.5):
            """
            判断当前位置是否到达目标点附近
            :param position: 当前车辆位置 np.array([x, y])
            :param target: 目标点位置 np.array([x, y])
            :param threshold: 距离阈值，默认0.5以内算到达
            :return: True 表示到达，False 表示未到达
            """
            distance = np.linalg.norm(position - target)
            return distance < threshold
        success = is_at_target(self.position, self.target)
        # 到达目标 只要小车距离目标小于 0.5，就认为到达目标，给一个大的正奖励 + 结束回合。
        if success:
            reward += 100
            done = True
        elif done and not success:
            reward -= 100  # 提前失败扣分

        return np.concatenate([self.position, self.velocity]), reward, done, {}

    def render(self, mode="human"):
        plt.clf()  #清空当前图像，避免每一帧都叠加。
        plt.xlim(0, 10)   #设定画布坐标范围（与你的状态空间一致）。
        plt.ylim(0, 10)

        # 绘制障碍  用 plt.Rectangle() 添加障碍区域。add_patch() 是 matplotlib 中在画布上添加图形对象的方法。
        for (x_min, y_min), (x_max, y_max) in self.obstacles:
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, color='gray'))

        # 绘制轨迹
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], 'b.-')
        plt.plot(self.target[0], self.target[1], 'go')  # 目标点
        plt.plot(self.position[0], self.position[1], 'ro')  # 当前车辆位置
        plt.pause(0.01)
        #短暂暂停以更新画面，实现“动画”效果。如果不用 plt.show() 是因为你是连续多帧绘制，不想阻塞。