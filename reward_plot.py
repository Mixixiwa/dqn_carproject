#绘制奖励曲线
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
reward_list = np.loadtxt("rewards.txt")

plt.figure()
plt.plot(reward_list, label="Episode Reward", color='blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward Curve")
plt.legend()
plt.grid(True)
plt.savefig("reward_plot.png")
plt.show()