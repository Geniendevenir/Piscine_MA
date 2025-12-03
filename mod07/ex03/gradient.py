import numpy as np

def gradient(x, y, theta):
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

	m = y.shape[0]
	X_Prime = np.hstack((np.ones((m,1)), x))
	y_hat = X_Prime @ theta

	error = y_hat - y

	gradient = (1 / m) * (X_Prime.T @ error)
	return gradient

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8] ])

y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))

# Example 1:
print(repr(gradient(x, y, theta1)))
# Output:
#array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])

# Example 2:
theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
print(repr(gradient(x, y, theta2)))
# Output:
#array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])