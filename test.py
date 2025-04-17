from car_env import CarEnv
from dqn_agent import DQNAgent
import torch
import matplotlib.pyplot as plt

# 初始化环境和智能体
env = CarEnv()
agent = DQNAgent(state_dim=4, action_dim=5)

# 加载训练好的模型参数
agent.model.load_state_dict(torch.load("dqn_model.pth"))
agent.model.eval()  # 设置为评估模式
agent.epsilon = 0.0  # 完全贪婪策略

# 可视化设置
plt.ion()  # 开启交互模式

# 测试若干轮
for ep in range(3):
    state = env.reset()
    total_reward = 0

    print(f"\n[测试 Episode {ep+1}]")
    for t in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        total_reward += reward

        if done:
            print(f"✔ 结束于 step {t+1}, 累计奖励: {total_reward:.2f}")
            break
    else:
        print(f"⚠ 未能成功结束，累计奖励: {total_reward:.2f}")

# 测试结束后关闭图形窗口
plt.ioff()
plt.show()
