import numpy as np
import pandas as pd


def load_data():
    # 准备数据
    iris = pd.read_csv('../data/iris.csv')
    print("总个数", len(iris))
    # 取其中两个字段最为变量 petal length (cm), petal width (cm)
    X = iris.iloc[:, [2, 3]].values
    name = iris.iloc[:, 4]

    # 将name值转数字化 Iris-setosa为-1，否则值为1
    y = np.where(name == "Iris-setosa", 0, 1)
    return X, y
