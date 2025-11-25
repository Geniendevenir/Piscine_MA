import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	if y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None
	result = (1 / y.shape[0]) * pow(y_hat - y, 2)
	return float(np.sum(result))

def rmse_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	if y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None

	return sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	if y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None
	result = (1 / y.shape[0]) * abs(y_hat - y)
	return float(np.sum(result))

def r2score_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	if y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None

	a = sum(pow(y_hat - y, 2))
	b = sum(pow(y - np.mean(y), 2))
	result = 1 - (a / b)
	return float(sum(result))



# Example 1:
x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

# Mean-squared-error
## your implementation
print(mse_(x,y))
## Output:
#4.285714285714286
## sklearn implementation
print(mean_squared_error(x,y))
## Output:
#4.285714285714286

# Root mean-squared-error
## your implementation
print(rmse_(x,y))
## Output:
#2.0701966780270626
## sklearn implementation not available: take the square root of MSE
print(sqrt(mean_squared_error(x,y)))
## Output:
#2.0701966780270626

# Mean absolute error
## your implementation
print(mae_(x,y))
# Output:
#1.7142857142857142
## sklearn implementation
print(mean_absolute_error(x,y))
# Output:
#1.7142857142857142
# R2-score

## your implementation
print(r2score_(x,y))
## Output:
#0.9681721733858745
## sklearn implementation
print(r2_score(x,y))
## Output:
#0.9681721733858745