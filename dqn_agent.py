#深度 Q 网络智能体
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):   #神经网络结构
    def __init__(self, input_dim, output_dim):  #构造函数 初始化网络结构。input_dim 是输入维度，比如你的小车状态 [x, y, vx, vy] 是 4 维。output_dim 是输出动作数，比如你动作空间有 5 个动作，所以是 5。
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )  #这是一个 三层的全连接前馈神经网络：输入层：input_dim -> 128（比如 4 -> 128）
           # 隐藏层：128 -> 128，ReLU 激活
            #输出层：128 -> output_dim（比如 128 -> 5）
#           每个输出代表一个动作的 Q 值，强化学习会从中选最大值的那个动作。

    def forward(self, x):
        return self.fc(x)  #将输入状态 x 传入 self.fc 神经网络中，输出所有动作的 Q 值。

class DQNAgent:  #智能体
    def __init__(self, state_dim, action_dim):  #构造函数
        self.model = DQN(state_dim, action_dim) #model 是主 Q 网络，进行动作选择和训练。
        self.target_model = DQN(state_dim, action_dim) #target_model 是目标 Q 网络，用于稳定目标值。。
        self.target_model.load_state_dict(self.model.state_dict()) #两者初始参数相同
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) #使用 Adam 优化器训练主网络。初始学习率0.001
        self.memory = deque(maxlen=10000)  #经验回放缓存（存储状态转移元组），最多存 10000 条。
        self.gamma = 0.95 # 折扣因子
        self.batch_size = 64 # 每次训练时的样本数量
        self.epsilon = 1.0 # 初始探索率
        self.epsilon_decay = 0.995 # 每步训练后探索率衰减
        self.epsilon_min = 0.05  # 探索率下限

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)  #有一定概率随机探索。
        state = torch.FloatTensor(state).unsqueeze(0) #将一个单个状态（4 维）转换成形状 [1, 4] 的张量，以便送入神经网络预测。
        with torch.no_grad():  # 关闭梯度计算。
            return torch.argmax(self.model(state)).item() #从输出的 Q 值中选择 值最大的动作索引。.item()把张量转换为普通 Python 标量（int），方便后续使用。

    #储存经验
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #将经历的状态转移元组存入经验池。

    def train(self):
        if len(self.memory) < self.batch_size:
            return   #若经验数量不足，不训练。

        batch = random.sample(self.memory, self.batch_size)  #从记忆中随机采样 batch。
        states, actions, rewards, next_states, dones = zip(*batch)

        #转成 tensor 类型用于网络计算。
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions) #计算当前 Q 值 从 Q 值中选出我们实际采取的动作对应的 Q 值。
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0] #让 目标网络 输入下一状态 next_states，预测下一个状态所有动作的 Q 值。
                                                                                #max(1)：选取下一个状态中 Q 值最大的那个，作为我们期待的「最优收益」。
                                                                                #keepdim=True 保留维度，让结果形状仍为 [batch_size, 1]。
        target_q = rewards + self.gamma * next_q_values * (1 - dones) #dones：布尔值，表示是否 episode 结束（终止状态）。

        loss = nn.MSELoss()(q_values, target_q)  #使用 均方误差损失（MSELoss）
        self.optimizer.zero_grad()  #每次 backward() 前都要把之前的梯度清零，否则梯度会累加（PyTorch 默认行为）。不清零的话，会导致参数更新错误，模型发疯。
        loss.backward() #反向传播，计算每个参数的梯度。
        self.optimizer.step() #用计算好的梯度对模型参数进行更新。

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    #将主网络参数复制到目标网络，常用于每隔几步同步一次。
