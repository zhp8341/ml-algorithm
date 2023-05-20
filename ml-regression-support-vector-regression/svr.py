import numpy as np
import joblib
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml
from sklearn.svm import SVR

if __name__ == '__main__':
    boston = np.genfromtxt('../data/boston.csv', delimiter=',', skip_header=1)
    print("总个数", len(boston))
    # x根据特征名称选取对应数据
    x = boston[:, 0:13]
    y = boston[:, 13]

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print("X_train个数", len(X_train))
    print("X_test个数", len(X_test))

    # 初始化模型
    model = SVR(kernel='linear', C=1.0, epsilon=0.1)
    # 模型训练
    model.fit(X_train, y_train)

    w = model.coef_
    b = model.intercept_

    print("权重向量 w: ", w)
    print("截距 b: ", b)

    # 模型预测
    y_pred = model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.2f}")
