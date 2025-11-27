import numpy as np
from fit import fit_

def predict(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		return None
	if x.size == 0 or theta.size == 0:
		return None
	if x.ndim != 2:
		return None
	if theta.shape == (2,):
		theta.reshape(2, 1)
	if theta.shape != (2, 1):
		return None

	# Add Intercept
	m = x.shape[0]
	X_prime = np.hstack((np.ones((m,1)), x))  # shape (m, 2)

	# Matrix multiplication
	y_hat = X_prime @ theta  # shape (m, 1)

	return y_hat

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1]).reshape((-1, 1))

# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
print(repr(theta1))
# Output:
#array([[1.40709365],
#[1.1150909 ]])

# Example 1:
print(repr(predict(x, theta1)))
# Output:
#array([[15.3408728 ],
#[25.38243697],
#[36.59126492],
#[55.95130097],
#[65.53471499]])