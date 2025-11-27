import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def cost(theta0, theta1, x, y):
    m = len(x)
    y_hat = theta0 + theta1 * x
    return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)


y = np.array([13.0, 12.5, 5.0, 15.0, 19.0, 16.5, 17.5, 21.5, 24.0, 30.0])

x = np.array([0.5, 2, 3, 4, 5, 6, 7, 8, 9, 10])

theta_0 = 0

theta1_vals = np.linspace(-3, 7, 400)
cost_vals_theta1 = np.array([
    cost(theta_0, t1, x, y) for t1 in theta1_vals
])
#plt.plot(theta1_vals, cost_vals_theta1)


theta_1 = 0
theta0_vals = np.linspace(-3, 7, 400)
cost_vals_theta0 = np.array([
    cost(t0, theta_1, x, y) for t0 in theta0_vals
])

plt.plot(theta0_vals, cost_vals_theta0)

plt.show()