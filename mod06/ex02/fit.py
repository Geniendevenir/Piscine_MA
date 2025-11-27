import numpy as np

def fit_(x, y, theta, alpha, max_iter):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
		return None
	if x.size == 0 or y.size == 0 or theta.size == 0:
		return None
	if x.ndim != 2 or y.ndim != 2:
		return None
	if x.shape[1] != 1 or y.shape[1] != 1 or x.shape != y.shape:
		return None

	if not isinstance(theta, np.ndarray):
		return None

	if theta.shape == (2,):
		theta = theta.reshape(2,1)

	if theta.shape != (2, 1):
		return None

	if not isinstance(alpha, float) or not isinstance(max_iter, int):
		return None

	theta = theta.astype(float)
	m = x.shape[0]

	for _ in range(max_iter):
		#PREDICT
		X_Prime = np.hstack((np.ones((m,1)), x))
		y_hat = X_Prime @ theta

		#EVALUATE
		gradient = (1/m) * (X_Prime.T @ (y_hat - y))

		#IMPROVE
		theta -= alpha * gradient

	return theta





