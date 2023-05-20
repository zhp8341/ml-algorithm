import numpy as np
from scatter_img import show_img
from load_data import load_data


# 手动写的逻辑回归模型 使用了梯度下降法来最小化逻辑回归模型的损失函数

# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义逻辑回归模型
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True):
        self.lr = lr  # 学习率
        self.num_iter = num_iter  # 迭代次数
        self.fit_intercept = fit_intercept  # 是否拟合截距
        self.theta = None  # 模型参数

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def cost_function(self, X, y, theta):
        m = X.shape[0]
        h = sigmoid(X.dot(theta))
        cost = -1 / m * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
        return cost

    def gradient(self, X, y, theta):
        m = X.shape[0]
        h = sigmoid(X.dot(theta))
        grad = 1 / m * X.T.dot(h - y)
        return grad

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            grad = self.gradient(X, y, self.theta)
            self.theta -= self.lr * grad

            if i % 10000 == 0:
                cost = self.cost_function(X, y, self.theta)
                print(f"迭代次数: {i}, 损失函数值: {cost:.4f}")

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


if __name__ == '__main__':
    # 准备数据
    X, y = load_data()

    # 训练模型
    # num_iter 迭代次数不一样预测结果不一样
    model = LogisticRegression(lr=0.1, num_iter=10000)
    model.fit(X, y)

    # 预测结果
    X_new = np.array([[5.0, 1.7], [1.4, 0.2], [1.3, 0.2]])
    y_pred = model.predict(X_new)

    print("预测结果：", y_pred)

    # 绘制决策边界
    show_img(X, y, model)
