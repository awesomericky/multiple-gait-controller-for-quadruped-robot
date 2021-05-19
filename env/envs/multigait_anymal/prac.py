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

update = "test"
start = 400
total_step = 150
single_step = 50

contact_log = np.load(f'contact_plot/contact_{update}.npz')['contact']

fig, ax = plt.subplots(1,1, figsize=(20,10))
img = ax.imshow(contact_log[:, start:start + total_step], aspect='auto')
x_label_list = [i*0.01 for i in range(start + single_step, start + total_step + 1, single_step)]
y_label_list = ['FR', 'FL', 'RR', 'RL']
ax.set_xticks([i for i in range(single_step, total_step + 1, single_step)])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
fig.colorbar(img)
ax.set_title("contact", fontsize=20)
ax.set_xlabel('time [s]')
plt.savefig(f'contact_plot/contact_prac_{update}.png')
plt.show()
plt.close()

# plt.colorbar()
# plt.title("contact", fontsize=20)
# plt.gca().axes.get_yaxis().set_visible(False)
# plt.xlabel("time [s]")
# plt.xticks(np.arange(start, start+log_step))
# plt.show()
# # plt.savefig(f'contact_plot/contact_{update}.png')
# plt.close()
