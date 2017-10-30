import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib import colors
from matplotlib import cm as cmx
from matplotlib import animation

# The animation of policies for a certain periond
ANIMATION_PERIOD = 30

data_dir = "./determinate_log"

file_name = 'latest_test.npy'

file_path = os.path.join(data_dir, file_name)

data = np.load(file_path)

shape = data.shape
period_num = shape[0] - 1
step_num = shape[1]
asset_num = shape[2] - 2

period_reward = []
for period in data:
    reward = 1.0
    for r in period[:, -1]:
        reward *= 1 + r
    period_reward.append(reward)
leverage = data[:, -1, -2]

# figure1
x_period = list(range(period_num))
# subplot 1
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = ax1.twinx()
# 注意这里的句柄一定要有逗号
line_period_reward, = ax1.plot(x_period, period_reward[1:], 'r', label='reward')
line_period_leverage, = ax2.plot(x_period, leverage[1:], 'b--', label='leverage')
ax1.legend(handles=[line_period_reward, line_period_leverage], loc=7)
ax1.set_xlabel('periods')
ax1.set_ylabel('reward')
ax1.set_title('reward at different period')
ax2.set_ylabel('leverage')

# subplot2
ax3 = fig1.add_subplot(2, 1, 2)
# best_index = np.argmax(period_reward)
best_index = len(period_reward) - 1
# best_index = 130

benchmark_reward = []
best_reward = []
reward = 1
for step in data[0]:
    reward *= 1 + step[-1]
    benchmark_reward.append(reward)
reward = 1
for step in data[best_index]:
    reward *= 1 + step[-1]
    best_reward.append(reward)
x_steps = list(range(step_num))
line_step_benchmark, = ax3.plot(x_steps, benchmark_reward)
line_step_bestTest, = ax3.plot(x_steps, best_reward)
ax3.legend([line_step_bestTest, line_step_benchmark], ['Best', 'Benchmark'], loc=4)
ax3.set_xlabel('steps')
ax3.set_ylabel('reward')
ax3.set_title('reward in a single period')

# figure 2
fig2 = plt.figure(2)
ax4 = fig2.add_subplot(1, 1, 1)
color_map = plt.get_cmap('jet')
norm = colors.Normalize(vmin=-0.03, vmax=0.03)
map_reward_change = ax4.pcolor(data[1:, :, -1], cmap=color_map, norm=norm)
plt.colorbar(map_reward_change, cmap=color_map, norm=norm)
ax4.set_xlim([0, step_num])
ax4.set_ylim([0, period_num - 1])
ax4.set_ylabel('periods')
ax4.set_xlabel('steps')
ax4.set_title('single step reward')

# figure 3
fig3 = plt.figure(3)
ax5 = fig3.add_subplot(1, 1, 1, xlim=(0, step_num), ylim=(0, 1))

values = range(asset_num)
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

line_list = ()

for line0 in list(range(asset_num)):
    colorVal = scalarMap.to_rgba(values[line0])
    line, = ax5.plot([], [], color=colorVal, label="Asset - " + str(line0 + 1))
    ax5.legend()
    line_list += (line,)


def init():
    for line_num in list(range(asset_num)):
        line_list[line_num].set_data([], [])
    return line_list


def animate(i):
    data_len = np.mod(i, step_num)
    for line_num in list(range(asset_num)):
        line_list[line_num].set_data(x_steps[0:data_len], data[ANIMATION_PERIOD, 0:data_len, line_num])
    return line_list


ANIMATION_PERIOD = best_index
animation0 = animation.FuncAnimation(fig3, animate, init_func=init, frames=step_num, interval=1)

plt.show()
