import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




if __name__ == '__main__':
    # 生成随机数据
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # 使用梯度下降法计算模型参数
    lr = 0.1  # 学习率
    n_iterations = 1000  # 迭代次数
    m = 100  # 样本数

    # 1 自己写的梯度下降
    theta = np.random.randn(2, 1)  # 随机初始化参数
    X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置项

    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr * gradients
    # 输出模型参数
    print(theta)



    # 2、使用sklearn的LinearRegression进行比较
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

    # 3、调用SGDRegressor
    sgd_reg = make_pipeline(
        StandardScaler(),
        SGDRegressor(max_iter=n_iterations, eta0=lr)
    )
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.named_steps['sgdregressor'].coef_)
