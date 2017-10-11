import numpy as np
import matplotlib.pylab as plt

def step_matlix(x):
    return np.array(x > 0, dtype=np.plt)

def sigmoid(x):
    return 0.5 * np.tanh(x) + 0.5

x = np.arange(-5.0, 5.0, 0.1)
y = step_matlix(x)
z = sigmoid(x)
plt.plot(x,y)
plt.plot(x,z)
plt.xlabel(r'$mu_0$')
plt.show()
