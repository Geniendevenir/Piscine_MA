import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
	if not isinstance(theta, np.ndarray) or theta.size <= 0:
		return None
	if theta.shape != (2,1) and theta.shape != (2,):
		return None
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
		return None
	if x.size <= 0 or x.ndim > 1:
		return None
	if y.size <= 0 or y.ndim > 1:
		return None

	m = x.shape[0]
	X_prime = np.column_stack((np.ones((m,)), x))  # shape (m, 2)
	y_hat = X_prime @ theta  # shape (m, 1)

	plt.scatter(x, y)
	plt.plot(x, y_hat.flatten())
	plt.show()

x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
#Example 1:
theta1 = np.array([[4.5],[-0.2]])
plot(x, y, theta1)	

# Example 2:
theta2 = np.array([[-1.5],[2]])
plot(x, y, theta2)

# Example 3:
theta3 = np.array([[3],[0.3]])
plot(x, y, theta3)

