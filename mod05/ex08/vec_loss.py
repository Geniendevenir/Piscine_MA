
import numpy as np

def loss_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.ndim != 1 or y_hat.ndim != 1:
		return None
	if y.shape[0] != y_hat.shape[0]:
		return None
	result = (1 / (y.shape[0] *2)) * pow(y_hat - y, 2)
	return float(sum(result))
