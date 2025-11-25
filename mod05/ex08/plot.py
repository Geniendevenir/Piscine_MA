import numpy as np
import matplotlib.pyplot as plt
from vec_loss import loss_

def plot_with_loss(x, y, theta):
	if not isinstance(theta, np.ndarray):
		return None
	if theta.size <= 0 or theta.ndim != 1 or theta.shape[0] != 2:
		return None
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
		return None
	if x.size <= 0 or x.ndim != 1:
		return None
	if y.size <= 0 or y.ndim != 1:
		return None
	if y.shape[0] != x.shape[0]:
		return None

	m = x.shape[0]
	X_prime = np.column_stack((np.ones((m,)), x))  # shape (m, 2)
	y_hat = X_prime @ theta  # shape (m, 1)
	#y_loss = loss_(y, y_hat)
	plt.plot(x, y_hat)
	plt.scatter(x, y)	
	for xi, yi, yi_hat in zip(x, y, y_hat):
		plt.plot([xi, xi], [yi, yi_hat], "r--")
	plt.show()
	

x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])
plot_with_loss(x, y, theta1)

# Example 2:
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

# Example 3:
theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)