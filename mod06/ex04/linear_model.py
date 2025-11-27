import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

df = pd.read_csv("are_blue_pills_magics.csv")
data = df.to_numpy()

x = data[..., 1].reshape(-1, 1)
y = data[..., 2].reshape(-1, 1)
thetas = np.array([[0], [1]])


lr = MyLR(thetas, max_iter=50000)

thetas = lr.fit_(x, y)

lr = MyLR(thetas)
y_hat = lr.predict_(x)
loss = lr.loss_(y, y_hat)

blue_cyan = np.array((0, 255, 255)) / 255

plt.scatter(x, y, c=blue_cyan)
plt.scatter(x, y_hat, c="#00FF00", marker="x", linewidths=2)
plt.plot(x, y_hat, c="#00FF00", ls="--")
plt.grid("on")
#plt.plot(x, y_hat.min)
plt.show()


