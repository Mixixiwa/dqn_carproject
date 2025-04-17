#启动训练并可视化
from car_env import CarEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
plt.ion()  # 开启交互模式
import torch
import time

env = CarEnv()  #创建环境对象 CarEnv()
agent = DQNAgent(state_dim=4, action_dim=5) #初始化 DQN 智能体：state_dim=4：状态是位置 (x, y) + 速度 (vx, vy)。action_dim=5：5 个动作（上、下、左、右、停止）。

episodes = 500  #设定训练轮数为 500。
reward_list = []  # 新增：记录每个 episode 的奖励

for ep in range(episodes):
    state = env.reset()   #重置环境，获取初始状态。total_reward 用来记录这一轮的累计回报。
    total_reward = 0

    for t in range(150):  #每个 episode 最多执行 150 步（防止跑太久）。
        action = agent.act(state)  #选择动作（ε-贪心）
        next_state, reward, done, _ = env.step(action)  #与环境交互
        agent.remember(state, action, reward, next_state, done) #存经验
        agent.train()  #训练网络
        state = next_state  #更新状态
        total_reward += reward

        if ep % 10 == 0:  #可视化（每10轮渲染一次）
            env.render()

        if done:  #提前结束 Episode
            break
    reward_list.append(total_reward)  # 每轮记录一次
    agent.update_target()  #每轮更新一次 target network，跟进主网络参数。
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min) #epsilon 每轮衰减，最小为 epsilon_min（如 0.05）。
    print(f"[Episode {ep + 1}] Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}") #输出训练信息

print("训练完成")
time.sleep(3)

# 在训练结束后添加
torch.save(agent.model.state_dict(), "dqn_model.pth")
print("模型已保存为 dqn_model.pth")  #保存模型参数为 .pth 文件，方便以后加载进行推理或测试。

with open("rewards.txt", "w") as f:
    for r in reward_list:
        f.write(f"{r}\n")


