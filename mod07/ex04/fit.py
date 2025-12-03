import numpy as np


def fit_(x, y, theta, alpha, max_iter):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
		return None
	if x.size <= 0 or x.ndim != 2:
		return None
	if y.size <= 0 or y.ndim != 2 or y.shape[0] != x.shape[0]:
		return None
	if not isinstance(theta, np.ndarray):
		return None
	if theta.size <= 0 or theta.ndim != 2 or theta.shape != (x.shape[1] + 1, 1):
		return None
	if not isinstance(alpha,float) or not isinstance(max_iter, int):
		return None

	m = y.shape[0]
	X_Prime = np.hstack((np.ones((m,1)), x))
	theta = theta.astype(float)

	for _ in range(max_iter):
		y_hat = X_Prime @ theta
		error = y_hat - y
		gradient = (1 / m) * (X_Prime.T @ error)
		theta -= alpha * gradient
	return theta

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
# Example 0:
theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
print(repr(theta2))
# Output:
#array([[41.99..],[0.97..], [0.77..], [-1.20..]])