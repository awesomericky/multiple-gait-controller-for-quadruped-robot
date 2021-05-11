import numpy as np
from scipy.optimize import curve_fit
import time

x = np.random.normal(size=(400,))
y = np.random.normal(size=(400,))

def sin(x, a, b, c, d):
    return a* np.sin(b*x + c) + d

start = time.time()
for i in range(400):
    param, param_cov = curve_fit(sin, x, y)
end = time.time()
print(end - start)
