import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit
# import time

# x = np.random.normal(size=(400,))
# y = np.random.normal(size=(400,))

# def sin(x, a, b, c, d):
#     return a* np.sin(b*x + c) + d

# start = time.time()
# for i in range(400):
#     param, param_cov = curve_fit(sin, x, y)
# end = time.time()
# print(end - start)

# # Figure 4

# # update = "test"
# update = 'hierarchy'
# start = 1700
# total_step = 200
# single_step = 50
# task_name = 'hierarchy_0.1_1.5'

# # contact_log = np.load(f'contact_plot/{task_name}/contact_{update}.npz')['contact']
# contact_log = np.load(f'raisimGymTorch/exp_result/exp2/contact_{update}.npz')['contact']
# # contact_log = np.log(contact_log + 1e-6)
# # contact_log = contact_log - np.min(contact_log)
# contact_log = np.where(contact_log > 0, 1, 0)

# fig, ax = plt.subplots(1,1, figsize=(20,5))
# img = ax.imshow(contact_log[:, start:start + total_step], aspect='auto', cmap='Blues')
# x_label_list = [i*0.01 for i in range(start + single_step, start + total_step + 1, single_step)]
# y_label_list = ['FR', 'FL', 'RR', 'RL']
# ax.set_xticks([i for i in range(single_step, total_step + 1, single_step)])
# ax.set_yticks([0, 1, 2, 3])
# ax.set_xticklabels(x_label_list)
# ax.set_yticklabels(y_label_list, fontsize=18)
# ax.set_xlabel('Time [s]', fontsize=12)
# plt.savefig(f'contact_plot/contact_prac_{update}.png')
# plt.show()
# plt.close()

# plt.colorbar()
# plt.title("contact", fontsize=20)
# plt.gca().axes.get_yaxis().set_visible(False)
# plt.xlabel("time [s]")
# plt.xticks(np.arange(start, start+log_step))
# plt.show()
# # plt.savefig(f'contact_plot/contact_{update}.png')
# plt.close()

period = 1
t_range = np.arange(0, 1, 0.01)
sin1 = np.sin(t_range * (2*np.pi / period))
sin2 = np.sin(t_range * (2*np.pi / period) + np.pi)
sin4 = np.sin(t_range * (2*np.pi / period) + np.pi)
# plt.plot(t_range, 0.5*sin1, color='green')
plt.plot(t_range, sin2, '--', color='red', linewidth=4)
plt.plot(t_range, 0.5*sin2, color='blue', linewidth=4)
plt.savefig('sin')
# plt.plot(t_range, sin3)
# plt.plot(t_range, sin4)
plt.show()





# # Figure 2

# change_step = 350
# previous_period = 1
# CPG_a_old = 2*np.pi / previous_period
# CPG_b_old = 0
# t_range = np.arange(0, 1, 0.001)
# previous_sin = np.sin(t_range * CPG_a_old + CPG_b_old)
# plt.plot(t_range[:change_step], previous_sin[:change_step], color='blue', label='T')
# plt.plot(t_range[change_step:], previous_sin[change_step:], '--',color='blue')

# next_period = 0.5
# CPG_a_new = 2 * np.pi / next_period
# CPG_b_new = 0
# current_sin_before = np.sin(t_range * CPG_a_new + CPG_b_new)
# plt.plot(t_range, current_sin_before, '--', color='red', label='T+k (before)')

# next_period = 0.5
# CPG_a_new = 2 * np.pi / next_period
# CPG_b_new = ((CPG_a_old - CPG_a_new) * (change_step * 0.001)) + CPG_b_old
# current_sin_after = np.sin(t_range * CPG_a_new + CPG_b_new)
# plt.plot(t_range[change_step:], current_sin_after[change_step:], color='red', label='T+k (after)')
# plt.xlabel('Time [s]')
# plt.legend(loc="lower right")
# plt.savefig('figure2')
# plt.show()