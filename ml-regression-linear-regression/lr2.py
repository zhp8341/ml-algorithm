### 使用sklearn编写一个线性回归 ，一个变量
### 预测一下出生人口

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    points = np.genfromtxt('/Users/edy/git/ml-algorithm/data/renkou.csv', delimiter=',', skip_header=1)

    # 提取points中的两列数据，分别作为x，y
    x = points[:, 0]
    y = points[:, 1]

    ### 通过最小二乘法求解w,b
    lr = LinearRegression()
    x_new = x.reshape(-1, 1)
    y_new = y.reshape(-1, 1)
    lr.fit(x_new, y_new)
    w = lr.coef_[0][0]
    b = lr.intercept_[0]
    print("w is: ", w)
    print("b is: ", b)

    y_pred = lr.predict(x_new)
    mse = mean_squared_error(y_new, y_pred)
    print('预测误差MSE:', mse)

    # 预测一下 2023 ,2024
    x_pred = np.array([[2023], [2024]])
    print(lr.predict(x_pred))
