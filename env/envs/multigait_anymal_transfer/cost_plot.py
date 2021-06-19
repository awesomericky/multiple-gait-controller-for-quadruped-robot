import numpy as np
import matplotlib.pyplot as plt

# Cost plotting

# a = 0.3
# a_1 = [a]
# for _ in range(1000):
#     a = a ** 0.997
#     a_1.append(a)

# a = 0.3
# a_2 = [a]
# for _ in range(1000):
#     a = a ** 0.998
#     a_2.append(a)

a = 0.3
a_3 = [a]
n_iter = 4000
for _ in range(n_iter):
    a = a ** 0.999
    a_3.append(a)

# plt.plot(range(1001), a_1, label='0.997')
# plt.plot(range(1001), a_2, label='0.998')
plt.plot(range(n_iter + 1), a_3, label='0.999', color='red')
plt.title('k_c = 0.3, k_d = 0.999')
plt.xlabel('Number of iteration')
plt.savefig('cost_scale')
plt.show()

# # Discount factor plotting

# n_step = 400
# control_dt = 0.01

# def discount_result(discount_factor, n_step):
#     result = [1]
#     for _ in range(n_step):
#         result.append(result[-1] * discount_factor)
#     return result

# t_range = np.arange(n_step + 1) * control_dt
# result = discount_result(0.97, n_step)
# plt.plot(t_range, result, label='0.97')
# result = discount_result(0.98, n_step)
# plt.plot(t_range, result, label='0.98')
# result = discount_result(0.985, n_step)
# plt.plot(t_range, result, label='0.985')
# result = discount_result(0.99, n_step)
# plt.plot(t_range, result, label='0.99')
# result = discount_result(0.996, n_step)
# plt.plot(t_range, result, label='0.996')
# plt.legend()
# plt.show()