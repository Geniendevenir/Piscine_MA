import numpy as np
from prediction import predict_

def loss_elem_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	if y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None
	result = pow(y - y_hat, 2)
	return result

def loss_(y, y_hat):
	lelem = loss_elem_(y, y_hat)
	if lelem is None:
		return None
	result = ((1 / (y.shape[0] * 2)) * sum(lelem))
	return float(result.flatten()[0])

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(repr(loss_elem_(y1, y_hat1)))
# Output:
#array([[0.], [1], [4], [9], [16]])

# Example 2:
print(loss_(y1, y_hat1))
# Output:
#3.0

x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
theta2 = np.array(np.array([[0.], [1.]]))
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

# Example 3:
print(loss_(y2, y_hat2))
# Output:
#2.142857142857143

# Example 4:
print(loss_(y2, y2))
# Output:
#0.0