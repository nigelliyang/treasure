import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib as mpl

data_dir = "./determinate_log"

file_name = 'sv2.npy'

file_path = os.path.join(data_dir, file_name)

data = np.load(file_path)

shape = data.shape
period_num = shape[0]-1
step_num = shape[1]
asset_num = shape[2]-2

period_reward = []
for period in data:
    reward = 1.0
    for r in period[:,-1]:
        reward *= 1+r
    period_reward.append(reward)
leverage = data[:,-1,-2]

# figure1
x_period = list(range(period_num))
# subplot 1
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(2,1,1)
ax2 = ax1.twinx()
# 注意这里的句柄一定要有逗号
line_period_reward, = ax1.plot(x_period, period_reward[1:], 'r', label = 'reward')
line_period_leverage, = ax2.plot(x_period, leverage[1:], 'b--', label = 'leverage')
ax1.legend(handles = [line_period_reward, line_period_leverage], loc=7)
ax1.set_xlabel('periods')
ax1.set_ylabel('reward')
ax1.set_title('reward at different period')
ax2.set_ylabel('leverage')

# subplot2
ax3 = fig1.add_subplot(2,1,2)
best_index = np.argmax(period_reward)
benchmark_reward = []
best_reward = []
reward = 1
for step in data[0]:
    reward *= 1+step[-1]
    benchmark_reward.append(reward)
reward = 1
for step in data[best_index]:
    reward *= 1+step[-1]
    best_reward.append(reward)
x_steps = list(range(step_num))
line_step_benchmark, = ax3.plot(x_steps, benchmark_reward)
line_step_bestTest, = ax3.plot(x_steps, best_reward)
ax3.legend([line_step_bestTest, line_step_benchmark], ['Best', 'Benchmark'], loc=4)
ax3.set_xlabel('steps')
ax3.set_ylabel('reward')
ax3.set_title('reward in a single period')

#figure 2
fig1 = plt.figure(2)
ax4 = fig1.add_subplot(1,1,1)
cmap = plt.cm.jet
norm = mpl.colors.Normalize(vmin=-0.03, vmax=0.03)
map_reward_change = ax4.pcolor(data[1:,:,-1], cmap=cmap, norm=norm)
plt.colorbar(map_reward_change,cmap=cmap, norm=norm)
ax4.set_xlim([0,step_num])
ax4.set_ylim([0,period_num-1])
ax4.set_ylabel('periods')
ax4.set_xlabel('steps')
ax4.set_title('single step reward')

plt.show()

# ax2 = fig1.add_subplot(2,1,2)

# sns.tsplot(data=period_reward)

# h = plt.plot(leverage)
# plt.show()
# for i_plot in [0, 1, 2, 3, 4, 5, 6]:
#     data_i = data[:, i_plot]
#     h = plt.plot(data_i, label="Shares - " + str(i_plot + 1))

# plt.legend()
# plt.show()