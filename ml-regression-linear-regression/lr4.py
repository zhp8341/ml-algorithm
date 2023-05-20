import numpy as np
import joblib
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml

#  数据说明文件说明
# CRIM - 各镇的人均犯罪率。
# INDUS - 每个城镇的非零售商业用地比例。
# CHAS - 查尔斯河虚拟变量（=1，如果区块与河流相连；否则为0）。
# NOX - 一氧化氮的浓度（每1000万份）。
# RM - 每个住宅的平均房间数。
# AGE - 1940年以前建造的自建房的比例。
# DIS - 到波士顿五个就业中心的加权距离。
# RAD - 辐射状高速公路的可达性指数。
# TAX - 每10,000美元的财产税全额税率。
# PTRATIO - 各镇的学生-教师比率。
# B - 1000(Bk - 0.63)^2，其中-Bk是各镇黑人的比例。
# LSTAT - 人口中地位较低的百分比。
# MEDV - 业主自住房屋的中位价值，单位为1000美元。

if __name__ == '__main__':
    # 读取数据集 (通过 load_boston 函数读取波士顿房价数据集)
    # 数据会从网站https://www.openml.org/ 拉取到本地
    # boston = fetch_openml(name='boston')
    # print(boston)

    boston = np.genfromtxt('../data/boston.csv', delimiter=',', skip_header=1)
    print("总个数", len(boston))
    # x根据特征名称选取对应数据
    x = boston[:, 0:13]
    y = boston[:, 13]

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print("X_train个数", len(X_train))
    print("X_test个数", len(X_test))
    print("X_train列数", X_train.shape[1])

    # 初始化模型
    model = LinearRegression()

    # 模型训练
    model.fit(X_train, y_train)


    w = model.coef_
    b = model.intercept_

    print("权重向量 w: ", w)
    print("权重向量 w列数: ", w.shape[0])
    print("截距 b: ", b)

    # 模型预测
    y_pred = model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.2f}")

    # 导出模型到本地文件中
    joblib.dump(model, '../export_model/boston-model.pkl')

    # 从文件中加载模型 并且预测一组数据
    model = joblib.load('../export_model/boston-model.pkl')
    pred_x = np.array([[1.05393, 0.00, 8.140, 0, 0.5380, 5.9350, 29.30, 4.4986, 4, 307.0, 21.00, 386.85, 6.58]])
    pred_y = model.predict(pred_x)
    print(pred_y)
