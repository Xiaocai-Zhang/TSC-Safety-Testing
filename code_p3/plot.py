import numpy as np
import matplotlib.pyplot as plt



y_values = np.load('./rewards/cumulative_reward_history_MP.npy')
window_size = 30
n = len(y_values)
smoothed_rewards = np.zeros(n)
for i in range(window_size):
    smoothed_rewards[i] = np.mean(y_values[:i+1])
for i in range(window_size, n):
    smoothed_rewards[i] = np.mean(y_values[i-window_size+1:i+1])
plt.figure(figsize=(8, 4))
plt.plot(smoothed_rewards)
plt.title('Plot of Accumulated Reward over Episodes (Smoothed)')
plt.xlabel('Episode')
plt.ylabel('Accumulated Reward')
plt.grid(True)
plt.savefig('./figs/plot_of_rewards_smoothed.png', dpi=500, bbox_inches='tight')
# plt.show()
print(max(smoothed_rewards))