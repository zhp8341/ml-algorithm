### 手动编写一个线性回归 ，一个变量
### 预测一下出生人口

import  numpy as np
import matplotlib.pyplot as plt


# 损失函数计算（均方误差）
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
    # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2
    return total_cost / M


# 先定义一个求均值的函数
def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num


# 定义核心拟合函数 （最小二乘法）
def fit(points):
    M = len(points)
    x_bar = average(points[:, 0])

    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)
    b = sum_delta / M

    return w, b

##预测函数
def prediction(w,b,x):
   print(w * x + b)



if __name__ == '__main__':
    points = np.genfromtxt('/Users/edy/git/ml-algorithm/data/renkou.csv', delimiter=',',skip_header=1)


    # 提取points中的两列数据，分别作为x，y
    x = points[:, 0]
    y = points[:, 1]

    # 用plt画出散点图
    plt.scatter(x, y)
    plt.show()

    w, b = fit(points)

    print("w is: ", w)
    print("b is: ", b)

    cost = compute_cost(w, b, points)

    print("损失值cost is: ", cost)

    plt.scatter(x, y)
    # 针对每一个x，计算出预测的y值
    pred_y = w * x + b

    plt.plot(x, pred_y, c='r')
    plt.show()

    prediction(w,b,2023)
    prediction(w,b,2024)