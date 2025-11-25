import numpy as np

def predict_(x, theta):
	# 2. Build X' by adding a column of ones
	m = x.shape[0]
	X_prime = np.column_stack((np.ones((m,)), x))  # shape (m, 2)

	# 3. Matrix multiplication
	y_hat = X_prime @ theta  # shape (m, 1)

	return y_hat

